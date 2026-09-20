/* ============================================================
 * AINode "Join a cluster": the whole browser side of joining, in one file.
 *
 * Config > Cluster gets one card that takes a master address and a join token
 * minted on that master (`ainode cluster token`) and POSTs them to
 * /api/cluster/join-self. That route does the join in-process and restarts
 * nothing, so this card's job after a success is to say which keys were written
 * and which command applies them.
 *
 * It replaces the browser onboarding wizard, which was 19 KB of first-run pages
 * that no deployed node could reach and that never wrote a cluster key even when
 * reached (#208). This is deliberately NOT in app.js: app.js is 320 KB and being
 * rewritten in parallel, and a card this self-contained has no reason to live
 * inside it. app.js holds exactly one line that mounts this.
 *
 * The decisions are pure functions (buildPayload, describeResult, joinCommand) so
 * tests/test_join_flow.py can run them under node with no DOM, the way auth.js is
 * tested. Requests go through AINodeAuth.fetch: /api/cluster/join-self is behind
 * the API key like every other route on this node.
 * ============================================================ */

(function (global) {
  'use strict';

  function esc(value) {
    return String(value === null || value === undefined ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  var AINodeJoin = {

    // -- decisions, all pure -------------------------------------------------

    /**
     * The request body, or {error} when the form is not ready to send.
     * Trims everything, because a pasted token arrives with a newline on it
     * often enough to be worth handling rather than reporting.
     */
    buildPayload(input) {
      var src = input || {};
      var host = String(src.host || '').trim();
      var token = String(src.token || '').trim();
      if (!host) return { error: 'Enter the master address, e.g. 10.0.0.1:3000' };
      if (!token) {
        return { error: 'Enter a join token. Mint one on the master: ainode cluster token' };
      }
      var body = { host: host, token: token };
      var name = String(src.name || '').trim();
      if (name) body.name = name;
      var iface = String(src.iface || '').trim();
      if (iface) body.interface = iface;
      if (src.allowMismatch) body.allow_version_mismatch = true;
      return { body: body };
    },

    /** The line an operator runs to apply a join. One home for the text. */
    joinCommand() {
      return 'sudo systemctl restart ainode';
    },

    /**
     * What to tell the operator about an answer from /api/cluster/join-self.
     * Returns {ok, tone, message, detail}. A version mismatch is its own tone,
     * because it is the one failure with a second thing the operator can do.
     */
    describeResult(status, body) {
      var data = body || {};
      if (status === 200 && data.ok) {
        var keys = (data.written || []).join(', ');
        return {
          ok: true,
          tone: 'success',
          message: 'Joined cluster ' + (data.cluster_id || '?') + '.',
          detail: 'Wrote ' + (keys || 'the cluster keys') + '. Nothing else changed. '
            + (data.signed_discovery
              ? 'Discovery on this cluster is signed. '
              : 'The master has no cluster secret, so discovery is unauthenticated. ')
            + 'Apply it with: ' + this.joinCommand(),
        };
      }
      var message = '';
      if (data.error && data.error.message) message = String(data.error.message);
      if (status === 409) {
        return {
          ok: false,
          tone: 'mismatch',
          message: 'Refused: the master runs a different AINode release.',
          detail: message || 'Update one side so both match, or tick "join anyway".',
        };
      }
      if (status === 403) {
        return {
          ok: false, tone: 'error',
          message: 'The master refused the token.',
          detail: message || 'Tokens expire and work once. Mint a fresh one.',
        };
      }
      return {
        ok: false, tone: 'error',
        message: 'The join did not go through' + (status ? ' (HTTP ' + status + ')' : '') + '.',
        detail: message || 'Check the master address and that it is reachable from here.',
      };
    },

    // -- the card ------------------------------------------------------------

    /** The card's markup. Ids are namespaced so nothing in app.js collides. */
    cardHtml() {
      var html = '';
      html += '<div class="config-card" id="join-cluster-card">';
      html += '  <h3 class="config-card-title">Join a cluster</h3>';
      html += '  <p class="config-card-desc">Mint a token on the master with '
            + '<code>ainode cluster token</code>, then paste it here. This writes '
            + 'the cluster id, the shared secret, the master address and the '
            + 'discovery port, and touches nothing else.</p>';
      html += '  <div class="config-secret-input-row">';
      html += '    <input class="form-input" id="join-host" placeholder="master address, e.g. 10.0.0.1:3000">';
      html += '  </div>';
      html += '  <div class="config-secret-input-row">';
      html += '    <input class="form-input" id="join-token" placeholder="join token" autocomplete="off">';
      html += '    <button class="config-btn" id="join-submit">Join</button>';
      html += '  </div>';
      html += '  <label class="config-card-desc" style="display:block">'
            + '<input type="checkbox" id="join-allow-mismatch"> '
            + 'Join even if the master runs a different release</label>';
      html += '  <div id="join-status" class="config-card-desc" hidden></div>';
      html += '</div>';
      return html;
    },

    /** Append the card to a mount element and wire its one button. */
    mount(mountEl, options) {
      if (!mountEl || typeof document === 'undefined') return null;
      if (mountEl.querySelector('#join-cluster-card')) return null;
      var wrapper = document.createElement('div');
      wrapper.innerHTML = this.cardHtml();
      var card = wrapper.firstChild;
      mountEl.appendChild(card);
      var self = this;
      var opts = options || {};
      var button = card.querySelector('#join-submit');
      if (button) {
        button.addEventListener('click', function () {
          self.submit(card, opts);
        });
      }
      return card;
    },

    _status(card, tone, message, detail) {
      var box = card.querySelector('#join-status');
      if (!box) return;
      var color = tone === 'success' ? 'var(--nvidia-green, #76b900)'
        : (tone === 'pending' ? 'var(--text-muted, #888)' : '#ef4444');
      box.hidden = false;
      box.innerHTML = '<strong style="color:' + color + '">' + esc(message) + '</strong>'
        + (detail ? '<br>' + esc(detail) : '');
    },

    /** Read the form, call the route, report the answer. */
    async submit(card, options) {
      var opts = options || {};
      var built = this.buildPayload({
        host: (card.querySelector('#join-host') || {}).value,
        token: (card.querySelector('#join-token') || {}).value,
        allowMismatch: (card.querySelector('#join-allow-mismatch') || {}).checked,
      });
      if (built.error) {
        this._status(card, 'error', built.error);
        return;
      }
      var button = card.querySelector('#join-submit');
      if (button) button.disabled = true;
      this._status(card, 'pending', 'Joining…');
      var status = 0;
      var body = null;
      try {
        var resp = await global.AINodeAuth.fetch('/api/cluster/join-self', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(built.body),
        });
        status = resp.status;
        body = await resp.json().catch(function () { return null; });
      } catch (e) {
        status = 0;
        body = { error: { message: String(e && e.message ? e.message : e) } };
      }
      if (button) button.disabled = false;
      var outcome = this.describeResult(status, body);
      this._status(card, outcome.tone === 'success' ? 'success' : 'error',
                   outcome.message, outcome.detail);
      if (outcome.ok) {
        var tokenInput = card.querySelector('#join-token');
        if (tokenInput) tokenInput.value = '';
        if (typeof opts.onJoined === 'function') opts.onJoined(body);
      }
    },
  };

  global.AINodeJoin = AINodeJoin;
  // Loadable by a node test without a browser.
  if (typeof module !== 'undefined' && module.exports) module.exports = AINodeJoin;
})(typeof window !== 'undefined' ? window : globalThis);
