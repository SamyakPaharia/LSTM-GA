// Static page structure. Inlined by the build so the first frame is complete; injected by
// main.js when the dev server serves an empty shell.

export const MARKUP = `
<div id="pc-root">
  <canvas id="pc-canvas" aria-label="Cricket ground"></canvas>

  <div id="pc-hud" hidden>
    <div class="hud-tl"><div id="sb" class="sb"></div></div>
    <div class="hud-tr">
      <div id="ticker" class="ticker"></div>
      <div id="speed" class="speed" hidden></div>
      <canvas id="radar" class="radar" width="150" height="150" aria-label="Field placings"></canvas>
    </div>
    <div class="hud-top">
      <div id="banner" class="banner" hidden></div>
      <div id="freehit" class="badge" hidden>Free hit</div>
      <div id="status" class="status" hidden></div>
    </div>
    <div id="flash" class="flash"></div>
    <div id="timing" class="timing"></div>
    <div id="meter" class="meter" hidden>
      <div class="meter-track"><div id="meter-sweet" class="meter-sweet"></div><div id="meter-fill" class="meter-fill"></div><div id="meter-mark" class="meter-mark" hidden></div></div>
      <span class="meter-lbl">Release</span>
    </div>
    <div class="hud-bottom">
      <div id="comm" class="comm" hidden></div>
      <div id="hint" class="hint"></div>
    </div>
    <div class="hud-keys">
      <button type="button" id="btn-card"><kbd>Tab</kbd>Scorecard</button>
      <button type="button" id="btn-cam"><kbd>V</kbd>Camera</button>
      <button type="button" id="btn-menu"><kbd>Esc</kbd>Menu</button>
    </div>
    <div id="touch" class="touch" hidden>
      <div id="stick" class="stick"><div id="knob" class="knob"></div></div>
      <div id="tbtns" class="tbtns"></div>
    </div>
  </div>

  <div id="ov" class="overlay" hidden><div id="ov-panel" class="panel" role="dialog" aria-modal="true"></div></div>
  <div id="card" class="card" hidden></div>

  <section id="scr-title" class="screen">
    <div class="rail">
      <p class="eyebrow">Limited-overs cricket in your browser</p>
      <h1 class="wordmark"><span>Pavilion</span><span>Cricket</span></h1>
      <p class="lede">Aim your length, time your release, pick your shot. Play the computer or a friend online, twenty overs or two.</p>
      <label class="field"><span>Your name</span><input id="in-name" maxlength="16" autocomplete="nickname" placeholder="Captain"></label>
      <nav class="menu">
        <button type="button" class="menu-item" id="go-solo"><b>Play the computer</b><small>Quick match · 2 to 20 overs</small></button>
        <button type="button" class="menu-item" id="go-online"><b>Play a friend online</b><small>Private room with a code</small></button>
        <button type="button" class="menu-item quiet" id="go-howto"><b>How to play</b><small>Controls and rules</small></button>
      </nav>
      <p class="fine">Original game. Teams use invented players. Sound starts after your first click.</p>
    </div>
  </section>

  <section id="scr-setup" class="screen" hidden>
    <div class="rail wide">
      <button type="button" class="back" data-back>← Back</button>
      <h2>Quick match</h2>
      <div class="group"><h3>Your team</h3><div class="teams" id="pick-team"></div></div>
      <div class="group"><h3>Opponent</h3><div class="teams" id="pick-opp"></div></div>
      <div class="row2">
        <div class="group"><h3>Overs a side</h3><div class="seg" id="pick-overs"></div></div>
        <div class="group"><h3>Difficulty</h3><div class="seg" id="pick-level"></div></div>
      </div>
      <button type="button" class="btn primary big" id="solo-start">Walk out to the middle</button>
    </div>
  </section>

  <section id="scr-online" class="screen" hidden>
    <div class="rail">
      <button type="button" class="back" data-back>← Back</button>
      <h2>Play a friend</h2>
      <p class="lede" id="online-mode">Checking how this page can connect…</p>
      <div id="online-server" class="group" hidden>
        <h3>Game server</h3>
        <label class="field"><span>Server address</span><input id="in-server" placeholder="wss://your-server.example/ws" autocomplete="off"></label>
      </div>
      <div class="online-actions" id="online-actions">
        <div class="group">
          <h3>Start a room</h3>
          <p class="note">You get a five-letter code to send to your friend.</p>
          <button type="button" class="btn primary" id="on-create">Create room</button>
        </div>
        <div class="group">
          <h3>Join a room</h3>
          <div class="join"><input id="in-code" maxlength="5" placeholder="CODE" autocomplete="off" spellcheck="false" aria-label="Room code"><button type="button" class="btn" id="on-join">Join</button></div>
        </div>
      </div>
      <p class="error" id="online-error" hidden></p>
    </div>
  </section>

  <section id="scr-lobby" class="screen" hidden>
    <div class="rail wide">
      <button type="button" class="back" id="lobby-leave">← Leave room</button>
      <h2>Room <span class="code" id="lobby-code">-----</span> <button type="button" class="btn small" id="lobby-copy">Copy code</button></h2>
      <p class="note" id="lobby-note"></p>
      <div class="seats" id="lobby-seats"></div>
      <div class="group"><h3>Your team</h3><div class="teams" id="lobby-team"></div></div>
      <div class="group"><h3>Overs a side</h3><div class="seg" id="lobby-overs"></div></div>
      <button type="button" class="btn primary big" id="lobby-start">Start match</button>
      <p class="note" id="lobby-wait" hidden>Waiting for the host to start the match…</p>
    </div>
  </section>

  <section id="scr-howto" class="screen" hidden>
    <div class="rail wide">
      <button type="button" class="back" data-back>← Back</button>
      <h2>How to play</h2>
      <div id="howto-body"></div>
      <div class="group rules">
        <h3>Rules in brief</h3>
        <p>Each side bats once for the overs chosen. Ten wickets ends an innings. A bowler cannot bowl two overs in a row, and can bowl at most a fifth of the overs.</p>
        <p>Wides and no-balls cost a run and are bowled again. After a no-ball comes a free hit: you can only be run out.</p>
        <p>Batters are out bowled, caught, LBW or run out. Over the rope on the full is six, along the ground is four.</p>
      </div>
    </div>
  </section>
</div>
`;
