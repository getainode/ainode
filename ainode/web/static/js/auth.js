/* ============================================================
 * AINode auth: the ONE place the dashboard talks about who is calling.
 *
 * Two credentials, one wrapper. A PERSON signs in at the front door and rides a
 * session cookie (`ainode_session`, HttpOnly, sent by the browser itself on
 * same-origin requests); a PROGRAM sends `Authorization: Bearer <key>`. Nobody
 * pastes a key to use the dashboard any more (#261). Every request the UI makes
 * goes through AINodeAuth.fetch(), which attaches the key when one is stored,
 * stamps `X-AINode-Client: dashboard` so the server honours the cookie, and
 * reports a 401 to whoever registered onUnauthorized(). Before the wrapper
 * existed, enabling auth left the shell rendering and every panel 401ing, which
 * is why the fleet ran with auth off (#167).
 *
 * The session half is decisions plus requests, no markup: loadMe() reads
 * /api/auth/me, bootDecision() turns that into "show the app" or "show the
 * sign-in screen", and signIn()/signOut() are the two calls the screen makes.
 * static/js/signin.js draws it.
 *
 * Deliberately dependency-free and DOM-free so it can be exercised under node
 * with a stub fetch and a stub storage (tests/test_auth_usable.py runs it that
 * way): the storage and the fetch it uses are both injectable, every storage
 * access is wrapped (a browser with site data blocked throws on read), and
 * nothing here touches document.
 * ============================================================ */

(function (global) {
  'use strict';

  // Namespaced and spelled out: it shows up in a devtools storage pane next to
  // the chat history, and "key" on its own would not say which key.
  var STORAGE_KEY = 'ainode.apiKey';

  // The CSRF rule, both halves of it in one place. A session cookie is sent by
  // the browser on every same-origin request, including one a third-party page
  // triggered, so the server only honours the cookie when the request also
  // carries this header: a form post or an <img> from another origin cannot set
  // a header, and a cross-origin fetch that tries needs a preflight this node
  // does not answer. Every request the dashboard makes carries it.
  var CLIENT_HEADER = 'X-AINode-Client';
  var CLIENT_NAME = 'dashboard';

  // What an operator runs on the node when auth is on and no account exists.
  // One home for the text: the sign-in screen and the 409 message both read it.
  function userAddCommand(name) {
    var who = String(name || '').trim();
    return 'ainode auth user add ' + (who || '<name>') + ' --admin';
  }

  function defaultStorage() {
    try {
      return global.localStorage || null;
    } catch (e) {
      // Accessing localStorage itself throws when site data is blocked.
      return null;
    }
  }

  var AINodeAuth = {
    STORAGE_KEY: STORAGE_KEY,
    CLIENT_HEADER: CLIENT_HEADER,
    CLIENT_NAME: CLIENT_NAME,
    userAddCommand: userAddCommand,

    // Injectable for tests. In a browser these are localStorage and fetch.
    storage: defaultStorage(),
    fetchImpl: (typeof global.fetch === 'function') ? global.fetch.bind(global) : null,

    // The last /api/auth/me payload: {user, auth_enabled, has_users}. null means
    // nobody has asked yet, or this node is old enough not to answer the route.
    me: null,

    // Set on the last 401 seen, so a panel opened afterwards can say what failed.
    lastUnauthorized: null,
    _handler: null,
    // The key as typed, for the case where storage is unavailable (a private
    // window, site data blocked): it works for this page load and is gone on a
    // reload, which beats refusing to use a key the operator just pasted.
    _memoryKey: '',

    // -- the stored key ----------------------------------------------------

    getKey() {
      try {
        if (this.storage) {
          var stored = this.storage.getItem(STORAGE_KEY);
          if (stored) return stored;
        }
      } catch (e) {
        /* fall through to the in-memory copy */
      }
      return this._memoryKey || '';
    },

    hasKey() {
      return this.getKey() !== '';
    },

    setKey(key) {
      var value = (key === null || key === undefined) ? '' : String(key).trim();
      if (!value) return this.clearKey();
      try {
        if (this.storage) this.storage.setItem(STORAGE_KEY, value);
      } catch (e) {
        // A private window with no quota: the key still works for this page
        // load, it just will not survive a reload. Nothing to report here.
      }
      this._memoryKey = value;
      return true;
    },

    clearKey() {
      try {
        if (this.storage) this.storage.removeItem(STORAGE_KEY);
      } catch (e) {
        /* nothing to undo */
      }
      this._memoryKey = '';
      return true;
    },

    /** The key as shown in a UI: enough to recognise, not enough to reuse. */
    maskKey(key) {
      var k = key === undefined ? this.getKey() : (key || '');
      if (!k) return '';
      if (k.length <= 8) return k.slice(0, 2) + '...';
      return k.slice(0, 4) + '...' + k.slice(-4);
    },

    // -- headers -----------------------------------------------------------

    /**
     * The caller's headers plus Authorization when a key is stored, and always
     * `X-AINode-Client: dashboard`. Pure: it returns a new object and never
     * mutates what it was given, and it never overwrites an Authorization the
     * caller set itself.
     *
     * The client header rides on GETs as well as writes. It costs nothing, and a
     * header that is sometimes there is a header somebody has to reason about.
     */
    headers(existing) {
      var out = {};
      var src = existing || {};
      // Accept a Headers instance as well as a plain object.
      if (typeof src.forEach === 'function' && typeof src.get === 'function') {
        src.forEach(function (v, k) { out[k] = v; });
      } else {
        Object.keys(src).forEach(function (k) { out[k] = src[k]; });
      }
      var hasAuth = Object.keys(out).some(function (k) {
        return k.toLowerCase() === 'authorization';
      });
      var key = this.getKey();
      if (key && !hasAuth) out.Authorization = 'Bearer ' + key;
      var hasClient = Object.keys(out).some(function (k) {
        return k.toLowerCase() === CLIENT_HEADER.toLowerCase();
      });
      if (!hasClient) out[CLIENT_HEADER] = CLIENT_NAME;
      return out;
    },

    // -- the wrapper every request goes through ----------------------------

    /**
     * fetch() with the credentials attached and a 401 reported once.
     *
     * Returns the response untouched, so callers keep their own status
     * handling: this adds headers and a notification, it does not swallow
     * anything. A rejected request (node down, offline) is reported as it was.
     *
     * `credentials: 'same-origin'` is what sends the session cookie. It is the
     * browser default for same-origin requests and set anyway, because the
     * dashboard's whole front door depends on it and a default is not a contract.
     *
     * `skip401Notify: true` keeps a 401 the caller EXPECTS from firing the
     * handler: a wrong password on the sign-in screen is answered on the screen,
     * not by dropping the person back onto it with a toast.
     */
    fetch(url, options) {
      var self = this;
      var opts = {};
      var src = options || {};
      Object.keys(src).forEach(function (k) { opts[k] = src[k]; });
      var quiet = opts.skip401Notify === true;
      delete opts.skip401Notify;
      opts.headers = this.headers(src.headers);
      if (!opts.credentials) opts.credentials = 'same-origin';
      var impl = this.fetchImpl;
      if (typeof impl !== 'function') {
        return Promise.reject(new Error('no fetch implementation'));
      }
      return impl(url, opts).then(function (resp) {
        if (resp && resp.status === 401 && !quiet) self._unauthorized(url, resp);
        return resp;
      });
    },

    // -- the session: who is at the keyboard -------------------------------

    /**
     * Read /api/auth/me. Open on purpose, so it answers before anybody signs in.
     *
     * null means "this node did not answer the question": a release older than
     * the front door has no such route, and the dashboard then behaves exactly
     * as it did before, on the API key alone. Never treat null as "signed out".
     */
    loadMe() {
      var self = this;
      return this.fetch('/api/auth/me', { skip401Notify: true }).then(function (resp) {
        if (!resp || !resp.ok) return null;
        return resp.json();
      }).then(function (body) {
        self.me = (body && typeof body === 'object') ? body : null;
        return self.me;
      }).catch(function () {
        self.me = null;
        return null;
      });
    },

    /** The signed-in person, or null when it is a key or an open node. */
    user() {
      return (this.me && this.me.user) ? this.me.user : null;
    },

    isAdmin() {
      var u = this.user();
      return !!(u && u.role === 'admin');
    },

    /**
     * What to render on load, from one /api/auth/me payload. Pure.
     *
     *   {show: 'app'}                      auth off, signed in, or a stored key
     *   {show: 'signin', state: 'form'}    auth on, nobody signed in
     *   {show: 'signin', state: 'no-accounts'}  auth on and this node has none
     *
     * A stored key still gets the app: a program's credential in a browser is
     * how the desktop app and a headless-node operator drive the dashboard, and
     * taking that away would strand them.
     */
    bootDecision(me) {
      var m = me || null;
      if (!m || !m.auth_enabled) return { show: 'app', state: 'open' };
      if (m.user) return { show: 'app', state: 'session' };
      if (this.hasKey()) return { show: 'app', state: 'key' };
      if (m.has_users === false) return { show: 'signin', state: 'no-accounts' };
      return { show: 'signin', state: 'form' };
    },

    /** True when the sign-in screen, not the app, is what this browser gets. */
    needsSignIn(me) {
      return this.bootDecision(me === undefined ? this.me : me).show === 'signin';
    },

    /**
     * The one-line error for a refused sign-in. Pure, so the wording is testable.
     * `retryAfter` is the Retry-After header's seconds, when the server sent one.
     */
    signInError(status, body, retryAfter) {
      var served = (body && body.error && body.error.message) ? String(body.error.message) : '';
      if (status === 429) {
        var wait = parseInt(retryAfter, 10);
        if (!isFinite(wait) || wait < 1) wait = 30;
        var unit = (wait === 1) ? ' second' : ' seconds';
        return 'Too many tries. Wait ' + wait + unit + ' and sign in again.';
      }
      if (status === 409) {
        return served || ('This node has no accounts yet. On the node, run: ' + userAddCommand());
      }
      if (status === 401 || status === 403) return served || 'Wrong name or password.';
      if (status === 400) return served || 'Enter your name and password.';
      return served || 'This node could not sign you in. Try again, and check the node is running.';
    },

    /**
     * POST /api/auth/login. Resolves to {ok: true, user} or
     * {ok: false, status, message, noAccounts}: the screen renders the message
     * and never has to know a status code.
     */
    signIn(name, password) {
      var self = this;
      var body = { name: String(name || '').trim(), password: String(password || '') };
      if (!body.name || !body.password) {
        return Promise.resolve({ ok: false, status: 0, message: 'Enter your name and password.' });
      }
      return this.fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        skip401Notify: true,
      }).then(function (resp) {
        var retryAfter = '';
        try { retryAfter = (resp.headers && resp.headers.get) ? resp.headers.get('Retry-After') : ''; } catch (e) { retryAfter = ''; }
        return resp.json().catch(function () { return null; }).then(function (payload) {
          if (resp.ok && payload && payload.user) {
            // The cookie is set. Re-read /api/auth/me rather than trusting the
            // login body, so one route is the authority on who is signed in.
            return self.loadMe().then(function () {
              return { ok: true, user: self.user() || payload.user };
            });
          }
          return {
            ok: false,
            status: resp.status,
            message: self.signInError(resp.status, payload, retryAfter),
            noAccounts: resp.status === 409,
          };
        });
      }).catch(function () {
        return {
          ok: false,
          status: 0,
          message: 'Could not reach this node. Check it is running, then try again.',
        };
      });
    },

    /**
     * POST /api/auth/logout. The stored API key is NOT touched: signing out ends
     * a person's session, and forgetting a program's key is a separate decision
     * the person makes in Config > API access.
     *
     * Logout is a write, so it sits behind the middleware like any other: a
     * caller with no credential left meets a 401 before the handler. That IS
     * signed out, so every answer ends the same way here, and the caller is told
     * only whether the node did the revoking.
     */
    signOut() {
      var self = this;
      return this.fetch('/api/auth/logout', { method: 'POST', skip401Notify: true })
        .then(function (resp) {
          self.me = null;
          return !!(resp && resp.ok);
        })
        .catch(function () {
          self.me = null;
          return false;
        });
    },

    /** Register the one handler that opens the API access panel. */
    onUnauthorized(fn) {
      this._handler = (typeof fn === 'function') ? fn : null;
    },

    _unauthorized(url, resp) {
      // hadSession says WHICH credential just failed, which is the difference
      // between "your session ended, sign in again" and "the key in this browser
      // is not one of this node's keys".
      this.lastUnauthorized = {
        url: String(url),
        at: Date.now(),
        hadKey: this.hasKey(),
        hadSession: !!this.user(),
      };
      if (!this._handler) return;
      // The dashboard fires a dozen requests per poll, so a 401 storm must open
      // the panel once, not a dozen times. 2s covers one poll cycle.
      var now = Date.now();
      if (this._lastNotifiedAt && (now - this._lastNotifiedAt) < 2000) return;
      this._lastNotifiedAt = now;
      try {
        this._handler(this.lastUnauthorized);
      } catch (e) {
        /* a broken handler must not break the request that triggered it */
      }
    },
  };

  global.AINodeAuth = AINodeAuth;
  // Loadable by a node test without a browser.
  if (typeof module !== 'undefined' && module.exports) module.exports = AINodeAuth;
})(typeof window !== 'undefined' ? window : globalThis);
