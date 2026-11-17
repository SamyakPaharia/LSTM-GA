# Pavilion Cricket — Product Requirements

## 1. Summary

Pavilion Cricket is an original 3D cricket game that runs in a web browser with no install.
It plays in the tradition of the classic PC cricket sims of the 2000s: you aim a pitch marker
and time a release meter to bowl, and you pick a footwork, a direction and a moment to swing
when you bat. You can play a full limited-overs match against the computer, or against a
friend online.

It is not a port or clone of any existing game. The name, interface, art, audio and squads
are original; player names are invented.

## 2. Goals and non-goals

### Goals

| # | Goal | How we know it is met |
|---|------|-----------------------|
| G1 | Play a complete match in the browser against AI | Toss to result with correct cricket scoring, at 60 fps on a mid-range laptop |
| G2 | Play a complete match against another person online | Two browsers, one room code, one batting and one bowling, innings swap, same scoreboard on both |
| G3 | Batting and bowling both take skill | Timing, footwork, shot direction, line, length and variation all change the outcome |
| G4 | Zero-install, zero-asset | No downloads beyond the page; all models, textures and sounds are generated in code |
| G5 | Multiplayer tolerates ordinary internet latency | No real-time position streaming; per-ball inputs only; playable at 200 ms round trip |

### Non-goals (v1)

- Manual fielding and manual catching (fielders are automatic).
- Test matches, tournaments, career mode, player editor.
- Licensed teams, real player names or likenesses.
- Spectators, matchmaking with strangers, ranked play, accounts.
- Anti-cheat beyond server-side rules and scoring (it is a game for friends).

## 3. Players and use cases

- **Solo player** — opens the page, picks two teams, overs and difficulty, and plays.
- **Two friends** — one creates a room and shares a five-letter code; the other joins. Each
  picks a team, the host picks the overs, and they play a full match with rematch at the end.

## 4. Game modes

| Mode | Description |
|------|-------------|
| Quick match vs AI | 2, 5, 10 or 20 overs a side. Easy / Medium / Hard. Human does the toss. |
| Online 1v1 | Private room by code. Host sets overs. Each player controls one team for the whole match: bats in one innings, bowls in the other. |
| How to play | Controls and rules reference inside the game. |

## 5. Match rules

- Limited overs, 11 players a side, 10 wickets, 6 legal balls an over.
- Toss: winner chooses to bat or bowl.
- A bowler may not bowl consecutive overs and may bowl at most ⌈overs ÷ 5⌉ overs.
- Extras: wide (+1, re-bowl), no-ball (+1, re-bowl, next ball is a free hit).
- Dismissals: bowled, caught (incl. caught behind), LBW, run out. On a free hit only run out counts.
- Boundaries: four if the ball bounces before the rope, six if it clears it.
- Strike changes on odd runs and at the end of each over. A new batter takes the dismissed batter's end.
- Second innings ends when the target is passed. Result is by runs, by wickets, or a tie.

## 6. Gameplay design

### 6.1 Bowling

1. Pick a bowler at the start of each over (eligible bowlers only).
2. Pick a delivery: pace bowlers have seam-up, outswinger, inswinger, cutter and slower ball;
   spinners have stock ball, arm ball, wrong 'un and flighted ball.
3. Move the pitch marker to choose line and length (bouncer to yorker).
4. Press once to start the run-up; a meter rises. Press again to release. The sweet zone gives
   full pace and accuracy; too low is slow and loose; over the top is a no-ball.
5. Choose a field preset (attacking, balanced, defensive, spin) before any ball.

Accuracy error and pace come from the bowler's rating and the meter. Swing curves in the air
toward the marker, so the early line misleads the batter. Cut and spin deviate off the pitch.

### 6.2 Batting

- The landing spot appears when the ball is released.
- Choose a direction (eight ways, relative to the screen), a footwork (front or back foot),
  optionally loft, or defend, or leave.
- **Timing** is the gap between your key press and the ideal moment: perfect, good, early,
  late or missed. Early drags the ball to leg and in the air; late slices it squarer or finds an edge.
- **Shot selection** is scored against the ball: front foot suits full balls, back foot suits
  short ones; hitting against the line is risky.
- Lofted shots clear the rope only when well timed; otherwise they hang in the air.
- You can shuffle across the crease before the ball.

### 6.3 Running and fielding

- After a hit, call each run yourself and send the batters back if it is not on.
- Fielders chase, catch and throw automatically. The throw goes to the end where a run out is
  most likely. Catches can be dropped; hard chances are dropped more often.

### 6.4 AI

- Bowls plans with realistic variety of length and line; accuracy scales with difficulty.
- Bats by reading line and length, finding gaps in the field, and adjusting aggression to the
  match situation (overs left, wickets in hand, required rate). Timing error scales with
  batter rating and difficulty.
- Runs only when the simulated throw leaves a safety margin.

### 6.5 Controls

| Situation | Keys |
|-----------|------|
| Bowling | Arrows: move marker · `Q`/`E`: delivery · `F`: field · `Space`: start run-up, `Space` again: release |
| Batting | Arrows: direction · `S`: front foot · `A`: back foot · `D`: defend · hold `Shift`: loft · `Z`/`C`: shuffle across crease |
| Running | `Space`: run · `X`: send back |
| Anywhere | `V`: camera · `Tab`: scorecard · `M`: mute · `Esc`: menu |

Touch devices get an on-screen stick and context buttons for the same actions.

## 7. Presentation

- 3D stadium: striped outfield, pitch with creases, stumps, rope, stands with crowd, floodlights, sky.
- Stylised low-poly players with procedural animation: run-up and delivery, stance and shots,
  running between wickets, fielding, catching, throwing, keeping, celebrations, umpire signals.
- Cameras: behind the batter when batting, behind the bowler's arm when bowling, and a high
  camera at the same end that follows the ball once it is hit.
- HUD: scoreboard, batter and bowler cards, this-over ticker, required rate, ball speed, timing
  feedback, field radar, commentary line, full scorecard.
- Audio synthesised in the browser: bat on ball, bounce, stumps, crowd swell, cheers.

## 8. Online requirements

| Requirement | Detail |
|-------------|--------|
| Room flow | Create → code → join → lobby (teams, overs) → start → toss → match → rematch |
| Authority | One host holds the match state and applies all rules and randomness |
| Latency model | Per-ball messages: delivery, shot, run calls, result. No streaming of positions |
| Fairness of view | The batter does not see the marker until release; the bowler's view runs slightly delayed so the opponent's swing appears in time |
| Idle handling | Auto-bowl / auto-pick after a timeout so one player cannot stall |
| Disconnects | Opponent is told; seat held for 60 s for reconnection; ball in progress is re-bowled; otherwise the match is forfeited |
| Hosting | (a) Node WebSocket server that anyone can deploy; (b) inside a Claude artifact, peer-to-peer through the artifact's live room with the creator's browser as host |

## 9. Non-functional requirements

- Desktop Chrome, Edge, Firefox and Safari (WebGL 1/2). Phone layout works with touch controls.
- Page weight under 1.5 MB including the 3D library; no images or audio files.
- 60 fps target at 1080p on integrated graphics; under 150 draw calls in play.
- No personal data collected. Nicknames are held in memory for the life of the room only.

## 10. Milestones

1. **Engine** — rules, delivery physics, shot model, fielding and running, AI. Verified by
   simulating hundreds of AI v AI matches and checking score distributions.
2. **Host and transports** — match host, room lobby, in-page link, WebSocket server.
3. **Client** — 3D scene, players and animation, cameras, HUD, input, audio, menus.
4. **Online** — lobby UI, WebSocket play, artifact room play, reconnect and timeouts.
5. **Packaging** — single-file build, artifact build, deployment files, docs.

## 11. Acceptance checklist

- [ ] A 5-over match vs AI runs from toss to result with a correct scorecard.
- [ ] Bowled, caught, LBW, run out, wide, no-ball and free hit all occur and score correctly.
- [ ] A perfectly timed lofted drive goes for six; a badly timed one can be caught.
- [ ] Two browser tabs complete a match through a room code with identical scoreboards.
- [ ] Killing one tab mid-match notifies the other and allows reconnection.
- [ ] AI v AI T20 simulations average a plausible total (roughly 130–190) and 5–9 wickets.
