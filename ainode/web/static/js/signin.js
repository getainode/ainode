/* ============================================================
 * AINode sign-in: the front door (#261).
 *
 * A person signs in here and stays signed in until they sign out. Nobody pastes
 * an API key to read a dashboard: keys are for programs, and the one place that
 * still takes one by hand is a collapsed block in Config > API access.
 *
 * A full page, not a modal. When this screen is up the app shell is not in the
 * document's flow and app.js has fetched nothing else: a node that wants a name
 * and a password must not show a shell full of panels that all 401 (#167 in a
 * new shape).
 *
 * The decisions live in AINodeAuth (bootDecision, signIn, signInError) so they
 * run under node with no DOM; this file is markup, focus, Enter, and the error
 * line. Same split as static/js/join.js.
 * ============================================================ */

(function (global) {
  'use strict';

  function esc(value) {
    return String(value === null || value === undefined ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  var AINodeSignIn = {

    // -- the markup, as one pure function -----------------------------------

    /**
     * The screen for a state. Pure, so tests can read the copy.
     *
     * state: 'form'         name and password, the normal case
     *        'no-accounts'  auth is on and this node has no accounts yet, so
     *                       there is nothing to type: it shows the one command
     *                       that creates the first one.
     * reason: a one-line explanation of why this screen is up (a revoked
     *         session, auth switched on mid-visit). Empty on a cold load.
     */
    screenHTML(opts) {
      var o = opts || {};
      var state = o.state === 'no-accounts' ? 'no-accounts' : 'form';
      var html = '';
      html += '<div class="signin-card">';
      html += '  <div class="signin-brand">';
      html += '    <img src="/static/img/logo.png" alt="" class="signin-logo">';
      html += '    <span class="signin-wordmark"><span class="signin-wordmark-ai">AI</span><span class="signin-wordmark-node">Node</span></span>';
      html += '  </div>';

      if (state === 'no-accounts') {
        html += '  <h1 class="signin-title">Sign in</h1>';
        html += '  <p class="signin-lead">This node requires a sign-in but has no accounts yet. Create the first one on the node, then reload this page:</p>';
        html += '  <pre class="signin-command"><code>' + esc(AINodeAuth.userAddCommand()) + '</code></pre>';
        html += '  <p class="signin-note">It asks for a password, stores it hashed in <code>~/.ainode/auth.json</code>, and takes effect with no restart.</p>';
        html += '  <button class="signin-submit" id="signin-reload">Reload</button>';
      } else {
        html += '  <h1 class="signin-title">Sign in</h1>';
        html += '  <p class="signin-lead">This node is private. Sign in to use the dashboard.</p>';
        html += '  <form class="signin-form" id="signin-form" autocomplete="on">';
        html += '    <label class="signin-label" for="signin-name">Name</label>';
        html += '    <input class="signin-input" id="signin-name" name="username" type="text" autocomplete="username" autocapitalize="none" spellcheck="false" required>';
        html += '    <label class="signin-label" for="signin-password">Password</label>';
        html += '    <input class="signin-input" id="signin-password" name="password" type="password" autocomplete="current-password" required>';
        html += '    <button class="signin-submit" type="submit" id="signin-submit">Sign in</button>';
        html += '  </form>';
      }

      html += '  <div class="signin-error" id="signin-error" role="alert"' +
              (o.reason ? '>' + esc(o.reason) : ' hidden>') + '</div>';
      html += '  <p class="signin-programs">Using this node from a program? It takes an API key (Config &gt; API access).</p>';
      html += '  <div class="signin-footer">' + this.TEXAS_MARK + 'Made in Texas</div>';
      html += '</div>';
      return html;
    },

    // The small Texas mark the dashboard footer already uses, so the front door
    // is signed the same way the rest of the UI is.
    TEXAS_MARK: '<svg class="tx-icon" viewBox="0 0 100 100" width="13" height="13" aria-hidden="true" focusable="false" style="vertical-align:-2px;margin-right:5px;fill:currentColor"><path d="M30 2h22v30h14l4 8 8 4 2 8 3 8 3 8-2 4-12 12-6 14-6-6-2-6-10-6-4-8-8-6-6-6-6-6-6-2-6-4-10-4v-4l28-8z"/></svg>',

    // -- showing it ----------------------------------------------------------

    _root() { return document.getElementById('signin-screen'); },
    _shell() { return document.getElementById('app-shell'); },

    /**
     * Put the screen up. `onSignedIn` is called once, after the cookie is set,
     * and is where app.js starts the app it did not start on load.
     */
    show(opts) {
      var o = opts || {};
      var root = this._root();
      if (!root) return;
      var self = this;
      this.onSignedIn = (typeof o.onSignedIn === 'function') ? o.onSignedIn : this.onSignedIn;
      this.state = o.state === 'no-accounts' ? 'no-accounts' : 'form';
      root.innerHTML = this.screenHTML({ state: this.state, reason: o.reason || '' });
      root.style.display = '';
      var shell = this._shell();
      if (shell) shell.style.display = 'none';
      document.body.classList.add('signin-mode');

      var reload = document.getElementById('signin-reload');
      if (reload) reload.addEventListener('click', function () { global.location.reload(); });

      var form = document.getElementById('signin-form');
      if (form) {
        // A form submit, so Enter in either field signs in and the browser's own
        // password manager sees a real login.
        form.addEventListener('submit', function (ev) {
          ev.preventDefault();
          self.submit();
        });
      }
      var name = document.getElementById('signin-name');
      if (name) name.focus();
    },

    hide() {
      var root = this._root();
      if (root) {
        root.innerHTML = '';
        root.style.display = 'none';
      }
      var shell = this._shell();
      if (shell) shell.style.display = '';
      document.body.classList.remove('signin-mode');
    },

    /** True while the front door is the page. */
    isUp() {
      var root = this._root();
      return !!(root && root.style.display !== 'none' && root.innerHTML !== '');
    },

    error(message) {
      var line = document.getElementById('signin-error');
      if (!line) return;
      if (!message) {
        line.hidden = true;
        line.textContent = '';
        return;
      }
      line.hidden = false;
      line.textContent = message;
    },

    submit() {
      var self = this;
      var nameEl = document.getElementById('signin-name');
      var passEl = document.getElementById('signin-password');
      var btn = document.getElementById('signin-submit');
      var name = nameEl ? nameEl.value : '';
      var password = passEl ? passEl.value : '';
      this.error('');
      if (btn) { btn.disabled = true; btn.textContent = 'Signing in...'; }
      return AINodeAuth.signIn(name, password).then(function (result) {
        if (btn) { btn.disabled = false; btn.textContent = 'Sign in'; }
        if (result.ok) {
          if (passEl) passEl.value = '';
          self.hide();
          if (typeof self.onSignedIn === 'function') self.onSignedIn(result.user);
          return true;
        }
        if (result.noAccounts) {
          // Nothing to type: swap to the state that says what to run.
          self.show({ state: 'no-accounts', reason: result.message });
          return false;
        }
        self.error(result.message);
        if (passEl) { passEl.value = ''; passEl.focus(); }
        return false;
      });
    },
  };

  global.AINodeSignIn = AINodeSignIn;
  // Loadable by a node test without a browser (the markup half is pure).
  if (typeof module !== 'undefined' && module.exports) module.exports = AINodeSignIn;
})(typeof window !== 'undefined' ? window : globalThis);
