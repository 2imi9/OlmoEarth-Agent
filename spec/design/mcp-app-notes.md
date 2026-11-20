# MCP App: why it works this way

These notes give the reason behind the MCP App's behaviour. What each panel does is in
[Usage.md](../Usage.md#in-claude-with-views-mcp-app); how to use it is in [mcp-app.md](../../docs/mcp-app.md); the words are in
[terms.md](../../docs/terms.md). Each note is one or two plain lines, under the same headings as Usage.md. The history of each
change is in [changes/1.5.0.md](../changes/1.5.0.md); the short release notes are in [CHANGELOG.md](../../CHANGELOG.md).

## Setup

- A panel is for something to look at or do. An intermediate step shows no card, so the chat is not filled with
  cards; its limits are in the text Claude reads and states.
- The panels' own tools are marked for the app only, so Claude cannot call them. The pictures' consent depends on
  this: only a press in the panel can turn pictures on.
- Claude Desktop runs one server for every chat, and MCP names no chat. So the run's record starts afresh on a key
  change and on `olmoearth_new_task`; otherwise one chat's calls would appear in another chat's report.

## Every panel

- Readers who were not experts did not know "window", "UTM zone" or "interval". The panels say square, point, part and
  range; the folds and the tools keep the technical words, so nothing is lost for an expert.
- A `WORDING:` comment marks each sentence with a panel word, so a word can be changed everywhere with one search.
- A card has at most two lines under its heading, all in the body colour, so every opening line is readable; the panel
  draws no more than two even if a tool ever sends three.
- Every card's fold has the same name, "Details", so a reader learns one word.
- Controls of one kind have one size, so a tick, a long word or a four-digit count never moves the layout.
- "Sent to the chat" is said in one place, under the button pressed, so a screen reader hears it once.
- One job at a time: a sent job turns the job buttons off, because a panel cannot take back a message that waits in
  Claude's message box. The host never tells the panel that the message was deleted there, so the user lifts the lock:
  "Choose another job" ends the card's line, and the focus then goes to the first job.
- A press that deletes something the app would have to download again ("Remove kept copies") takes two presses within
  six seconds; Escape, or the focus moved away, takes the first press back. A double-click is one press, and the armed
  words are also said in a line only a screen reader gets.
- "Copy it into the chat" (not "type it"): the box under it selects the whole message with one click.
- A message or an error sits under what caused it: the reply to a press under the press, a link's failure beside the
  link. At the card's foot it stood far below the press, and was sometimes said twice.
- The watchdog's line is styled by a few rules in each page's head, so it is readable even when the card's own style
  never loaded; before, it showed in default serif type on a white page.

## The start card

- The card's one line says only what to do next. What a check cannot tell is said on the Check card and the sheet,
  where labelling happens.
- Each job is one tile that holds its label and its line, so the line never sits loose beside or under its button.
- Signed out, a tile keeps its words in the readable secondary colour: the disabled grey was 1.9:1 on white. "Sign in
  to start a job." is at body size, since it is the only word that says the tiles are off.
- On a wide panel the jobs come before the figure, so the figure does not push the sign-in and the jobs down.
- On a narrow panel the docs figure's words would be too small to read, so a strip of three boxes says it in plain
  words. Its example map is a real finished Studio map from the docs, not a tile drawn in a palette of its own.
- The figure's box moves in the page, not by a CSS order, so a screen reader and Tab follow what is seen.
- "Request access on olmoearth.allenai.org" gives a person with no account a way to get one; "Do not use someone
  else's key" because a key is a person's own.

## Sign-in

- The key goes from the browser page straight to this server, never in an MCP message, so the Claude app never relays
  or logs it.
- The page answers only its own host and origin, so a page on another site, or a DNS-rebound name, cannot post a key.
- A typed key is checked with Studio before it is used or saved, so a wrong key is never kept.
- Every sign-in text says the same two plain lines. A first line that only said the heading again was cut, and so was
  "and is checked with Studio before it is used", since a wrong key shows its own error.
- The app's own key (set when the app was installed) has no Sign out: the panel cannot remove it, so the button would
  do nothing.

## The projects panel

- A field's hint is the text's colour at .7, 5.3:1 on white; at .6 it was 4.0:1, too faint.
- A date has the month's name from a fixed table, so no reader takes it for another day, whatever the browser's
  language.
- The card's one line says only what to press: the heading already says what the card is, and what a check cannot
  tell is said where labelling happens.
- A project with no description shows nothing: "No description" told the reader nothing.
- "Show outputs" names the button's action; "Outputs" read as a label.
- The outputs toggle has one fixed width, so every row's job buttons start at one left edge.
- The line under a prediction says what the map shows and whether the check can run, so a user does not start a check
  that can only fail. A class map has no such line: its Check button already says the check can run.
- With several outputs, the list says that Studio names them only by their ids and to ask whoever made the run: a
  result carries only its id and the names of what it holds, so the panel cannot tell which one is the user's.
- The list's first line is the one place that says the run made several maps; the status line and the row do not
  repeat it.
- While that list is open the row's job buttons are hidden, since their press would only open the list again.
- "Where to look first" is off on Studio results (`FIRST_IN_STUDIO`): no recorded experiment supports that order on a
  Studio result, and the grid reads only about 100 single pixels.
- "Check how wrong it is" is not offered for embeddings or detections: the check can only fail there.
- After a Sign out that leaves the app's own key, the user is still signed in, so the panel shows that key's projects
  and Claude is told the user is still signed in.
- "In progress" lists counts only, never a window, place, class or label. Sheets name no Studio account, so every
  account's sheets on this computer are listed.
- A glyph says what a thing is, never how good it is. Mint is kept for a completed status, so a glyph never uses it.
- What a project's ring shows is said once, above the cards; a caption beside every ring said it again and again.
- A model's record field names were read from live records, but their value spellings were not seen. A value in an
  unknown spelling is drawn as "not recorded", never guessed.
- The projects note puts the press first: the rows already say what each map outputs, so Claude need not list the
  predictions again.

## The Check card

- `clip_out_of_range`: real 0-to-1 scores run a little outside 0 to 1 (two real results ran from -0.0147 to 1.0070 and
  from -0.0865 to 1.781). Clipping counts those pixels instead of refusing the map. A score that reaches -1 or 2 is
  not a 0-to-1 score at all.
- The map is prepared in the background: a large map's first read took 53 s on an idle Mac and 147 s on a busy one,
  and Claude Desktop waits 180 s for a tool call. The call waits 20 s, then answers with a card that fills in by itself.
- The card hands Claude the map's text whole, outside the 800 characters of facts. Within them only the start would
  fit: a two-part map's say-first lines alone fill that room, and the note, `must_state` and `forbidden_claims` never
  fit.
- Claude calls `olmoearth_check_map` a second time only when the card could not hand on that text: in Claude Desktop
  each call draws a card, so a second call shows a second Check card of the same map.
- The planned counts are kept on disk: when they were kept only in memory, planning took 44 s of a 67 s reopen of a
  two-part map. Files older than 30 days are removed, so an app update's old files go.
- Step 3 says "what the map is meant to find". "Recognise it" pointed at nothing on the face, and "what the map shows"
  was read as the map's own answer at the square, the one thing a blind sheet hides.
- One line under the count question, "More squares: more work, and a narrower range.", replaces the same words under
  every count. Each count is its number alone, so counts never break unevenly on a narrow panel.
- A part's heading says "(1,234 squares in all)": "(300 squares)" over the counts read as a fourth count, or as the
  work to do.
- A chosen count has a tick, as a chosen answer on the sheet has, so the two chosen looks are the same.
- The strips picture says why a map is in parts. Its sentence is said only where it is true; other maps get the plain
  parts line.
- A part is named West, East, North or South only when that is certain, and never by its UTM zone, which a reader does
  not know.
- Each part's card is a section named by its heading, so a screen reader tells two parts' "Open the sheet" apart.
- Once a part is picked, "How this check works" folds, so the parts and their sheets come first.
- The line over the press says that the rule and the number are fixed at the press, and when the result comes. The
  come-back line ("Picked before? Press again: the same squares open.") is in Details, so the face keeps one line over
  the press.
- A press held by the rule or the box only moves the focus to the box: a toast said the line's words again right under
  it, and stayed after the rule was fixed.
- The finished heading says "In each part's sheet, press ...: one result per part": "Open a sheet" or "Open each
  part's sheet in turn" read as one press for the whole map's result.
- A failed pick says which press retries it ("Press Pick the squares to try again", "it waits for a later press"):
  "The press tries again" did not say which press, or whose.
- "The press got no answer back." is said in the page's hidden status, not a toast: the toast hung under another
  part's card and read as that part's line.
- On a card of parts the status line sits under the press or the open sheet, where the user acted; at the page's foot
  it was under both folds.
- "The sheet open here has a change that is not saved" is in the sheet's footer: as a toast it stayed under the other
  sheet, asking for a press of a Save labels no longer drawn.
- A sheet that cannot be opened names the button to press by its label; "Press the button" did not say which. Its red
  line shows only after a press fails, not over the button from the start.
- The server's reason, with a file name, is in the part's Details, so the face stays in plain words.
- The red card "The check could not start" says in plain words that nothing picked is lost and to ask Claude, who has
  the reason. Before, it was one red line in the tool's words (such as "GET /prediction-results/files failed").
- A card's error drops the exception's name, the OSError number and the hint written for Claude: they are for Claude,
  and Claude's text keeps them whole.
- A class map without top-k scores: its grid spans every zone, so the Check card cannot cut it into squares. Claude asks
  two plain questions and never says "design" or "budget", words readers did not know.
- That case is known by the package's contract, never by the error's words, so a rewording cannot break it.
- With no rule, the example sits in a small line under the box: as the box's grey placeholder, it went at the first
  key.
- The app's rule says "what the map is meant to find": "what the map shows" was read as the map's own answer, which a
  blind sheet hides.
- The app's rule is general for any map of presence and absence. The app does not guess what a map shows, so the box
  lets the user write the map maker's meaning, and Claude does not ask what the map is of.
- The box's label and its line name no map, since the rule is for any map of presence and absence.
- One map keeps its first rule: a map drawn or labelled under an earlier wording keeps that wording, so all its parts
  are judged by one rule.
- The 0-to-1 score is cut strictly above 0.5, as the pooling counts it. The cut is fixed, so no count the tools compute
  changes.
- A map of numbers gets two plain lines with no tool name, no "probability to pool" and no "threshold"; Claude's text
  keeps the tool's sentence, which names the tool Claude needs.
- "Preparing this map took too long" is the failed card's red card and says that nothing picked or saved is lost.

## The label sheet

- A class map's sheet has no right-or-wrong switch and no blind tick box: it never shows the map's answer, since labels
  given after seeing it pull the error rate down.
- The page gets handles, never a window's index, row, column, place, order or stratum, so the panel cannot show where
  a window is or how the map ranked it. The handle hides the window from the page, not from Claude, which reads the
  plan.
- A save refused because the sheet changed would be refused again, so the panel stops saving and offers "Reopen the
  sheet". Its footer says what happened and what to do, where a greyed "Not saved" button said neither.
- After its own removal of a set of answers fails to read the sheet again, the panel says the failure once, in the
  footer; the status line says only which answers were removed.
- The version is a keyed digest under the sheet's own secret, so a server restart keeps it; when two saves change the
  same square, the first save wins.
- The save lines sit in the sheet's footer, beside Save labels and the counter they are about, since a message sits
  under what caused it.
- Answers wait 1.5 s before a save, at most 5 s, so a run of answers saves once.
- Cannot tell is counted right at the range's lower end and wrong at its upper end, so the range covers both.
- The two-map switch stays locked until the result is asked for, so the squares are answered blind. Once the user has
  seen the maps' answers the sheet's answers lock, so the estimate reads the sheet as it was answered blind.
- The lead says the answers are locked; "without looking at the two maps" asked for answers the lock refuses. The
  switch's "(keep it on while you answer)" went once the switch was locked, since it asked for nothing.
- The finish button's reason ("Answer every square first.") stands beside it, so a grey button has its reason in words;
  a third lead line that said it was cut.
- "After you press, do not change an answer." says what matters once the sheet is ready; the counter and the saved note
  already say it is all answered and saved.
- A graded sheet's note says its answers are locked, since the panel locks them; why is in Details.
- A graded sheet's button reads "Show the result again": the job's label invited a new estimate. Where the answers
  changed since, it reads "See why the result does not count".
- A graded sheet with two sets of answers says only "This sheet now holds two sets of answers.": "so no result can come
  from it", under the note that the sheet gave its result, read as if that result did not exist.
- The rule stands right above the answers, in body type with "Rule:" in bold, so it is in view at each answer; its
  scope is in Details.
- A sheet with no rule says so on its face from the first open: the user must decide a rule before the first answer.
- Records keep their words: a sheet made with an earlier wording of the app's rule keeps it, and nothing is written
  again to say so.
- The cloud line is worded as a case ("If cloud covers the square on every date"), not as a fact about the square
  shown, so a square clouded on every date is pressed Cannot tell, not Absent.
- The example pictures' captions say "what the map is meant to find", for the same reason as the rule.
- A sheet with no pictures and no places ends with a question of its own, "No records? Stop and tell Claude.", so the
  way out is plain.
- The pictures come before the answers on the face, so the labeller looks before answering.
- Pictures and Street map are outlined buttons, so they look pressable. The pictures' line says what to press and that
  the user is asked first, where "Nothing to look at yet." said neither.
- The answer given has a pale fill, an outline, bold words and a tick, not the dark fill of the sheet's one action, so
  an answer never looks like the button that sends.
- A class name breaks after an underscore, not in mid-word.
- The dates wrap onto rows: a strip that scrolled sideways cut its next date at the panel's edge and took a keyboard
  stop of its own.
- A list row with no answer shows "(not labelled yet)": "(class seen)" looked like an answer.
- Street map is said once, in the open pane's head, not on every row.
- The focus margin is the footer's measured height plus 8 px: a fixed 140 or 170 px was smaller than the footer on a
  narrow panel, so the footer hid the focused control.

## Satellite pictures

- The pictures are the one exception to "no STAC fetches", on the condition that they stay off until the user switches
  them on, after a question that says what is sent and to whom.
- The question is three short lines, with the hosts and the details in a fold. The answer is kept with the words it
  answered, so a changed question is asked again.
- The fetch has its own HTTP client with no credentials, no redirects and no proxy, to exactly two https hosts, so
  it sends nothing but what the question names.
- Errors are fixed sentences with no coordinate, path, host or scene id, so a place never shows in an error.
- The dates are chosen by the same rule for every square, blind to its map class, stratum and order, so the pictures
  do not hint at the answer.
- The outline is yellow with black lines inside and outside, so it shows on cloud, sand and snow; it sits one pixel
  out, so no line covers the square.
- Scene cloud is in a fold: it is the whole image's cloud, not the square's, and would mislead on the face.
- The cloud rule (round r5-p). Each month's date is the scene with the least cloud and shadow on the outlined square
  by the scene's own Scene Classification Layer (SCL, made by ESA's Sen2Cor with every L2A product; a 20 m file on the
  same file store as the bands, so no new host and no change to the question). Covered classes: 3 (cloud shadow), 8 and
  9 (cloud of medium and of high probability), 10 (thin cirrus), the default of rslearn's `Sentinel2SCLToMask`. Snow
  (11), dark features (2) and unclassified (7) are not covered: they are often the real ground. Up to 6 masks are read
  per month, least scene cloud first, stopping at the first that marks nothing on the square; then up to 3 scenes'
  bands. When the scene its mask shows clear has too little data on the square, mask reading resumes before the next
  band read, so no scene is picked unscored while its mask can be read. A mask counts when it has data on
  at least half of the square. A date is "cloudy" on the pane (`cloud_here` "covered", folded) when the mask marks at
  least one pixel of the square. A mask with data on less than half of the square never counts as clear: the record
  keeps it as "covered", and when it marks nothing the pane shows its date as one with no mask ("unknown", not
  folded); its Details line says "no data on most of the square", with no share.
- The picture is checked too (20 Nov 2026, a coastal map's label sheets pictured live): Sen2Cor's SCL classed squares
  white with thick cloud right at a coastline as land and water (classes 5 and 6, the same counts as on a clear date,
  cloud probability 0), so the date picked was a white square. A scene whose picture is white on at least 25% of the
  outlined square (red, green and blue reflectance all at least 0.4: the missed cloud was 0.8 to 1.0; clear water,
  plants, sand, concrete and most beaches stay below) counts as cloudy whatever its mask says; the next
  scene is tried. When every scene tried is white there, the least cloudy by both signals is shown, folded, with the line
  "picture: white on N% of the square (cloud, fog, or bright ground such as snow)". The month keeps `white_share`; the pick is
  `scl-cover@2`.
- Cloudy dates are folded away, never dropped: the mask is sometimes wrong (bright roofs, sand, surf, snow), so the
  labeller can still open them. When every date is marked, all show and one line says so. A date with no mask is shown
  as before. The masks have their own small budget (15 seconds, 32 MiB, 250 requests): a mask that cannot be read never
  fails the pictures, and that date is ranked by scene cloud as before. The request cap was 150 before the round's
  review: 12 months of 6 masks are 72 masks, each up to 3 requests. The 15 seconds can still end the mask reads first
  on a cold call; a mask left unread falls back to the least-scene-cloud order (not yet measured on live masks).
- The masks of all the months are read first (up to 8 months at a time), then the bands: which months are scored does
  not depend on when a month's bands start. The time that pass took (at most 15 seconds) is added to the bands' 40
  seconds, so slow masks never make the bands late; the reads of one call can now take up to 55 seconds in all. When
  the band reads fail, the masks' time is ended too, so no mask read holds the failing call.
- Sources of the cloud rule: ESA SentiWiki, the Sentinel-2 L2A Scene Classification table (the class numbers); rslearn
  `Sentinel2SCLToMask` (the excluded classes 3, 8, 9, 10).
- A label read from these pictures is a reading of the map's own sensor, so the result is agreement with that reading,
  not ground truth; the result says so.

## The result card

- The headline's subject is the part, so the sentence cannot be passed on as the whole map's rate.
- The rate in words is the nearer of "1 in N" and "N in 10" inside the printed range: "about 1 square in 2" for 41%
  named a rate outside the card's own range.
- The caveat line ("Checked against satellite pictures ...") is at the size of the other text, not the card's last and
  smallest line under three figures.
- "This check says nothing about the area of any one class." is in Details, so the face does not carry a fifth
  sentence.
- "You did not see A in any square": "You saw A in no square" was read by a quick reader as the classes seen.
- The tally draws one row per kind of square: 4,000 squares drawn one by one were a wall several screens tall.
- The class count lines are in Details, so the face holds one range.
- Each part's words sit inside its own box, and a part is never narrower than its label on one line: a small part's
  label once wrapped to four lines and made the whole row as tall.
- "Check the other part too" says where to do it, "Ask Claude to open the Check card again", since nothing on this card
  checks a part.
- The parts are never averaged: the check gives no range for a combined rate.
- With Cannot tell the headline says "10.0% of squares, or 12.5% if the 2 Cannot tell were wrong", so "to" belongs to
  the range alone; "between 10.0% and 12.5%" was read as the range. The bar's label is "10.0% or 12.5%" for the same
  reason.
- A withdrawn result shows no rate on its face: answers changed after seeing where the map disagrees make the result
  invalid. A red line over the full headline still read the old rate as the result. Its line ends with the next step,
  "ask Claude for one".
- A new result needs another seed: the same seed and count open the locked sheet again.
- The unfinished line names the sheet's own finish button: the result card has no button of its own.
- The graded mark is tied to the design's content, so a lost design file written again keeps it; changes are told by a
  digest of the answer cells, so a spreadsheet's re-save with no answer changed is no change.

## The plan card

- The two lines say whose answers count, that the pick is random and fixed, and when the result comes. They do not
  repeat what the button and the sheet's greyed finish button already show.
- Once every square is answered, the second line names the next step and the button by its label in quotes.
- Once the sheet opens in the card, the second line goes: the sheet's footer says it.
- The card's one button sits right under the lines that lead to it, before Details.
- The face shows no file path: the CSV shows the map's class, so answering there is not blind.
- The plan's figure waits while the sheet is labelled blind: it shades each window by confidence, which on a
  confidence design is its stratum.
- Where panels show, Claude gets the plan without its list of windows: the host hands that text to the panel too, and
  the panel labels blind.

## Look-first

- A Studio result is not ranked by default: no recorded experiment supports the order on a Studio result, and the grid
  reads only about 100 single pixels.
- The first line says the number was chosen in advance, so it is not read as a count of doubtful squares.
- The grid picture is in Details: its own words say "windows", "review budget" and "review set", which no face says.
- The cards' box is six and a half rows tall, so a longer list shows half a card, not a row with its edge shaved.
- One Street map stands in the open pane, named for its square: one under every card repeated the same words outside
  its card.
- A card has one Details fold, and the numbers' fold sits inside it, not as a second fold.
- The check button sits right under its line, before Details, and the lead then has no second line, since the card
  says it with its button.
- The page gets an opaque key and a rank, never a path or a place.
- Street map buttons for points are outlined and one width: a bare digit was too small to press.
- A Street map failure is said under its own row of buttons: at the card's foot it was under Details.
- The CSV line tells the latest press only, so it never names an earlier file under a later failure. Every row carries
  "not a random sample; no error rate comes from it".

## The area map

- The card's one instruction is the first line under the map, at body size: as a small line at the bottom, a first-time
  user saw no instruction and dragged the map.
- No coordinates go to the chat and the panel shows none: the place stays in Studio, named by its id.

## What Claude is told

- Claude Code keeps only about the first 4,100 characters of a server's instructions, so the rules that matter most
  sit in that part.
- What a panel tells Claude is facts, not requests, and ids and counts only, so a panel never sends a place, a class or
  a label.
- Claude never calls a map "recommended": it does not know which map the user means.

## Files and privacy

- The output folder is made mode 700 and the kept files mode 600, so other users of the computer cannot read them.
- A Studio result's zip is kept for a day, so a Check card opened again needs no download.
- The pages load scripts only from pinned versions checked by hash, and each view's CSP names only those origins, so a
  changed script elsewhere cannot run in a panel.

## Limits

- The Check card's face holds more lines than the rule of two. Which of them goes is not decided, so the list stays in
  [Usage.md](../Usage.md#limits).
