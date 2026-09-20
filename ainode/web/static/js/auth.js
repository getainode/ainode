/* ============================================================
 * AINode API key: the ONE place the dashboard talks about auth.
 *
 * Every request the UI makes goes through AINodeAuth.fetch(), which attaches
 * `Authorization: Bearer <key>` when a key is stored and reports a 401 to
 * whoever registered onUnauthorized(). Before this existed, enabling auth left
 * the shell rendering and every panel 401ing, which is why the fleet ran with
 * auth off (#167).
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

    // Injectable for tests. In a browser these are localStorage and fetch.
    storage: defaultStorage(),
    fetchImpl: (typeof global.fetch === 'function') ? global.fetch.bind(global) : null,

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
     * The caller's headers plus Authorization when a key is stored. Pure: it
     * returns a new object and never mutates what it was given, and it never
     * overwrites an Authorization the caller set itself.
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
      return out;
    },

    // -- the wrapper every request goes through ----------------------------

    /**
     * fetch() with the key attached and a 401 reported once.
     *
     * Returns the response untouched, so callers keep their own status
     * handling: this adds a header and a notification, it does not swallow
     * anything. A rejected request (node down, offline) is reported as it was.
     */
    fetch(url, options) {
      var self = this;
      var opts = {};
      var src = options || {};
      Object.keys(src).forEach(function (k) { opts[k] = src[k]; });
      opts.headers = this.headers(src.headers);
      var impl = this.fetchImpl;
      if (typeof impl !== 'function') {
        return Promise.reject(new Error('no fetch implementation'));
      }
      return impl(url, opts).then(function (resp) {
        if (resp && resp.status === 401) self._unauthorized(url, resp);
        return resp;
      });
    },

    /** Register the one handler that opens the API access panel. */
    onUnauthorized(fn) {
      this._handler = (typeof fn === 'function') ? fn : null;
    },

    _unauthorized(url, resp) {
      this.lastUnauthorized = { url: String(url), at: Date.now(), hadKey: this.hasKey() };
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
