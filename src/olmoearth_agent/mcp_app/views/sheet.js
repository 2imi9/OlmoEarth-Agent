/* The label-sheet editor, shared by the sheet view, the result view and the
   Check card.

   data is olmoearth_app_sheet's answer (mcp_app/sheet.py read_sheet), the
   page copy: the sheet's kind and label rule, the face it is labelled on
   (the column this page writes: seen, class, wrong or pair), the answers
   that face takes, the counts, the job, and its windows. Each window comes
   as a handle the server keys with a secret of its own, with its answer on
   the face, in the order of the handles: the page never has a window's
   index, place, row, column, order or stratum, nor the map's class except
   on the face that labels right or wrong. Answers go back by handle through
   olmoearth_app_save_labels, with the sheet's id, the face and the label
   rule the page drew; only the windows the reviewer changed are sent.

   What the estimate reads (tools/estimation.py), and so what "every window
   is labelled" means here:
   - face 'seen' (a scores map whose design names its classes): the class
     seen, one of data.answers (Present, Absent, Cannot tell), on every
     window. The server writes the code into reference_class and works out
     wrong from the design's map class when each label is saved;
   - face 'wrong' (a scores map's, a class map's labelled right or wrong):
     wrong, 0, 1 or '?' (Cannot tell, bounded both ways by the estimate), on
     every window; the map's class is shown, since right or wrong needs it;
   - face 'class' (a class map): the class seen (reference_class) on every
     window; the server says when the other column holds labels that would
     make a mix the estimate refuses (data.mix_problem), and offers to clear
     that column;
   - face 'pair' (a two-map sheet): the class seen on every window, '?'
     where the reviewer cannot tell.
   The answers are the server's (data.answers, each with its keys): a
   certificate plan's sheet has no Cannot tell, so the panel offers none;
   a class map's class face offers it after its classes (the last round's
   S4: '?' in reference_class, which the estimate bounds both ways).

   Two ways to show the windows, one press apart ("How to show the squares",
   remembered for the editors drawn after it on this page, OE.sheetView;
   every window word the user sees is the sheet's unit word, "square" or
   "point", from OE.sheetWords):
   - One square, the default: the face (a section with tabindex -1). Its
     heading "Square k of n" is a live region, with Previous and Next; under
     it the window's Pictures and Street map and its pictures, then what the
     window shows beside its answer (the map's class on the right-or-wrong
     face, and a two-map sheet's pair once its blind switch is unticked) and
     the face's answers as buttons (r5-k K4: the pictures come before the
     answers), each with its key ("Present (P)"; with more than nine, the
     first nine, Cannot tell where it is taken, and a list of the rest; on a
     face that takes any class, "Other class..." to type one, the group's
     last cell, r5-e H3). One key
     listener on the face: an
     answer's key sets the answer (never clears it) and the next window not
     yet labelled shows, in the order of the handles, wrapping round; the
     left and right arrow keys move between windows, except inside a control
     that scrolls sideways, where they scroll it; modified, held and composing keys and
     keys typed into a field are left alone (a class typed with an IME
     commits its text on Enter, nothing more). A press on
     an answer sets it, or clears it when it is the answer given, as in the
     list. After an answer the focus stays on the face (a click inside it
     also focuses it), with a ring when the keyboard brought it there; with
     every window labelled, the finish button takes
     the focus once it is ready. Nothing focuses the face on load. Inline, the
     face is a pager with the dates side by side, wrapped onto as many rows
     as they need (r5-k K4; a horizontal strip before); in the host's full
     screen (.face.full, OE.sheetMode) the dates are a grid. Full screen is
     the header action the page passes (OE.fullscreenButton, shown only
     where the host offers full screen, and in full screen "Back to the
     chat": r4-k K5); the panel never asks for a mode by itself. While the pictures are open or on they show the window the face
     shows at once, from what the server gave for it or "Loading", and are
     asked for 300 ms after a move (a quick run of answers asks once), so
     the face never shows one window's pictures under another's heading.
   - List: the windows as rows. "#" is the window's place in this list, a
     button that shows that window on the face; then the map's class where
     the face shows it, the row's Show under the head Pictures (r5-e H4; Hide
     on the row whose pictures are open, and the button's name starts with
     the word it shows; on a sheet with places and no pictures, Street map
     instead), and the answer. With a row's pane open, its head holds that
     window's one Street map (r5-e H4). The filter (any, unlabelled,
     labelled) applies to the list only.
   r5-r (the owner, 20 Nov 2026: "a button to the next empty one, so we will
   not have this 99/100 scenario"): while a window has no answer, the face
   has a third button beside Previous and Next, "Next unlabelled" (N1: the
   next window with no answer after the one shown, wrapping round; the key U
   where no answer of the sheet has that key; one size with the other two,
   in equal columns that wrap to a second row on a narrow panel). While 1 to
   3 windows are left and at least one is answered, the counter's row names
   them, each a text button that shows it ("Unlabelled: Square 29"), and the
   grey finish button's line ends with a text button that does the same as
   N1 (N2). They show a window's place only, never what the map says.

   A class map's sheet has no switch of face (the owner's O1 of 14 Nov
   2026): it is labelled with the class seen, and the panel never shows the
   map's answer on it. A sheet already labelled right or wrong (by a
   spreadsheet) opens on the face the server offers first. A mix of the two
   label columns says the sheet holds two sets of answers, with a button
   that removes the set in the way, each named by what it holds (r5-b J21:
   "Remove the right-or-wrong marks, keep the classes picked"; the server's
   own sentence is in the Details fold); the press clears that column
   (clear_column), then reads the sheet again. While a column is
   cleared no answer is taken (the face or the list is inert and
   aria-busy), so none is lost when the editor is drawn again. A clear the
   server took whose sheet then cannot be read says the answers were
   removed, and the page offers to reopen the sheet.

   Pictures (olmoearth_app_window_image, by handle): the server finds the
   window from the files on disk and answers with pictures, the labelling
   rule and what a label then is, or, until the user switches pictures on,
   the question to ask (who receives what). The pictures never depend on the
   window's map class, so a blind sheet stays blind. The question card also
   offers a public test scene, unrelated to the sheet, so the user can see
   the fetch work before the sheet's extent is sent. Street map (reader test
   2's A/F21) asks the server for that one window's OpenStreetMap link at
   the press (olmoearth_app_window_link) and opens it; it never shows the
   map being checked.

   Pictures that fail or are missing (r5-a L1, L2). A picture call waits up
   to 200 s (OE.PICTURES_WAIT_MS). A call that throws shows one fixed
   sentence (OE.PICTURES_NOT_ARRIVED), never the thrown error's text, with a
   "Try again" button in the pane that repeats the call for that window; the
   server's own sentences another try can get past (its answer's again: a
   slow fetch, the catalogue, the image files) get the button too, the test
   scene's included. When no month of the map's period has a scene with data
   on the window (the answer's no_image), the pane says one fixed line
   (OE.NO_IMAGE) in place of a list of empty months, and on a face that
   takes Cannot tell, " Press Cannot tell.".

   Blind when labelling the class seen: the map's class is not sent, and no
   page is ever sent a stratum. The one blind switch is a two-map sheet's
   (the owner's O1 took the class map's away): unticked by the reviewer, it
   asks the server for the two maps' classes at that press
   (olmoearth_app_sheet with unblind; each row's list then leads with its
   pair, and the face says it). Ticked again, they are hidden; the reviewer
   has seen them. The switch is locked (r5-k K2, the owner's O2: ticked,
   aria-disabled, the line under its words says when it opens) until every
   window is answered and saved and the sheet's result was asked for (the
   finish press Claude took); an answer changed after that press, here or
   taken in from elsewhere, locks it again until a new press Claude takes.
   Right or wrong needs the map's class in view, so it cannot
   be blind and has no switch; nor has a seen sheet or a class map's sheet,
   which never show the map's answer.

   Each save sends the sheet's id, the face and the label rule the page
   drew; the server refuses a save once the sheet was drawn again or its
   rule changed (sheet.CHANGED_UNDER_PAGE), which cannot succeed on a retry:
   the page stops saving, says so, and offers "Reopen the sheet", which
   reads the sheet again and draws a new editor in place of this one (the
   changes not saved are not kept). Any other refusal invites a press of
   Save (or, once another sheet was opened in this editor's place, names the
   windows not saved). options.onSaved runs after each save the server took
   (the plan card reads its sheet's counts again then). The answers waiting
   for the 1.5 s save are sent at once when the panel is hidden or unloaded
   (best effort). A page that swaps this editor for another first awaits its
   oeSettle (save what is waiting; false when that failed, and the editor
   stays; true at once, with no save, once the sheet changed under the page)
   and may ask oeUnsaved how many windows' changes the swap drops; once the
   swap is done it calls oeLeave, which clears this editor's last toast
   about its sheet having changed under the page, while it is still shown
   and no change is left unsaved (r4-l L3). An editor replaced drops its
   pictures timer and page-wide listeners (oeDispose, OE.disposeEditors).
   When Claude closes the card (r4-k K1), the page's teardown awaits each
   editor's oeFlush: a quiet save of what is waiting (no toast, no onSaved,
   no focus move), bounded in time.

   One live copy of the sheet. The page copy carries the sheet's version
   (data.version, a keyed digest of the file the server made), and each save
   sends it; the server refuses a save once the sheet was saved elsewhere
   since this page read it (another panel, a spreadsheet: err.stale). The
   page then reads the sheet again on its face and takes it in (adopt): a
   window this page did not change takes the label saved there; a window it
   changed keeps its change where the label there is still the one this page
   last read or saved (state.base), and takes the label saved there where
   both changed it (the first save wins, and the toast names those windows
   by their place in this list, never by an index); then it sends what is
   left, with the new version. At most two such re-sends in a row: after
   them the page says the sheet keeps changing elsewhere and waits for a
   press of Save. A read that finds the sheet drawn again, its rule or this
   face no longer offered, stops saving as above. The page also asks the
   server for the version (olmoearth_app_sheet_progress, counts only) when
   it is shown again or gets the focus, at most once every 3 s and never
   while a save is under way or a class is being typed, and before the
   finish button sends its message (after any save, read or check under
   way): a sheet saved elsewhere is read and taken in, and a finish press
   that found a change, or could not read the sheet again, sends nothing, so
   the user sees it and presses again; a sheet that gave its result since
   this editor was drawn (Claude graded it after the finish press: the
   counts' graded) is read and drawn again, its answers locked (r5-k K3),
   once nothing here waits to be saved. A save whose answer says the server
   worked out more windows than were sent (rederived) is followed by a read,
   taken in with every change made here winning. Taking a sheet in keeps the
   window shown, the list's order, the focus where it was (OE.focusIfHere)
   and a class being typed.

   What Claude hears. The editor tells Claude silently (OE.tellClaude, its
   'sheet' part, where the host takes context updates) the sheet's name, its
   labels_csv_path and design_path, how many windows are labelled (with how
   many say Cannot tell), whether every change is saved, and once complete
   the finish button's words: when it is drawn, after each save, after a save
   made elsewhere is taken in and when it stops saving, never on each answer;
   never a window, a place, a class or a label.

   The footer (reader test 2's B/sheet-finish-button): the counter, then
   "Saved automatically" as plain text while nothing waits to be saved, or
   Save labels while a change waits, a save runs or one failed; and from the
   first open the finish button, named for the sheet's job (read_sheet's
   finish_label: estimate the error rate, certify the zone, or say which map
   is more accurate), greyed until every window is labelled and saved, then
   with "After you press, do not change an answer." beside it (A/F17, r5-b
   J29; r5-e H5: its first sentence, which said again what the counter and
   the saved note say, is gone; on a sheet that already gave its result,
   "This sheet already gave its result.", r5-d R06); its message
   (finish_message) names the tool
   Claude calls. After a press the footer says what became of the message
   (OE.sendJob, r5-b J20), as the job buttons do: OE.JOB_SENT once the host
   took it, or OE.JOB_NOT_SENT in the error colour over the whole text when
   it did not. Once the sheet changed under the page (r5-b J15) no Save
   button shows: beside the counter, how many answers given here are not
   saved ("2 answers not saved", or "Not saving"), and beside "Reopen the
   sheet", at body size, that the sheet was changed somewhere else, whether
   answers were lost and what to do; its answers are locked then (r5-k K3),
   as are a sheet's that already gave its result. The statuses of the
   sheet's saves are in the footer too (r5-k K5, the owner's O11), no
   longer in the status line at the page's foot: a save that went through
   ("Saved to ...") beside the counter in the place of "Saved automatically",
   which shows only while nothing waits; what Save labels causes while it is
   shown (a save that failed, "Not saved: ...", and its press with nothing
   waiting, "Every change is saved to ...") on a line under the row with
   Save labels; and a save made elsewhere taken in beside the counter it
   changed.

   The labelling step in the list. After an answer (a press or a key that
   sets one, a class picked from the list, a typed class accepted), the focus
   goes to the next window not yet answered, in the order shown, wrapping
   round: to its group of answers (tabindex -1; its list, for the class
   seen), never to an answer button, so Enter or Space alone answers
   nothing; with every window answered it goes to the finish button once
   that is ready. A press that clears an answer stays, and so does browsing a
   list with the arrow, Home, End or page keys or typed letters. Keys on the
   focused row set its answer, never clear it: the keys the server gives each
   answer of a seen or right-or-wrong face (P, A, ? and 1 to 9; R, W and ?,
   no digits, owner ruling 6); the class seen 1 to 9 for the n-th class of
   the row's list, and ? where the face takes Cannot tell (a two-map sheet, a
   class map's). A held key does not repeat: a repeated keydown answers
   nothing, and a repeated Enter or Space presses no button, the finish
   button included. A line in the Details fold names the keys of the view
   shown, and each answer button carries aria-keyshortcuts.
   The pictures follow the focus 300 ms after the answer when the pane is
   open or pictures are on (state.picturesOn); a closed pane with pictures
   off stays closed, and the server still asks before it fetches any
   picture. Where the panel is wide enough the open pane sits beside the
   table (.sheet-work.pane in common.js), else under it. Nothing takes the
   focus on load: every move goes through OE.focusIfHere. Rows, the face and
   their labels name a window by its place in this list ("square 3"), never
   by its index.

   The list's render() rebuilds the rows; the control that had the keyboard
   focus gets it back (by its data-focus key, else the same control in the row
   now at its place), so labelling with the keyboard does not lose its place.

   The reader tests' words (13 Nov 2026, OE.sheetWords), laid out by the last
   round's S1 (reader test 2: the face was a wall of text) and by round
   r5-b's J13 and J14 (the user-side audit: five paragraphs stood before the
   first answer, and the rule was the smallest text). Under "Label sheet",
   the part of the map it is for (data.part.heading, the server's); a graded
   sheet (data.graded) says, at body size, that its answers are locked
   (r5-k K3; it said they should not change now; r5-e H5: in two lines, its
   reason in the fold), and its lead
   keeps its first line only (J06); then the lead, two lines (r5-e H1; it
   was three): what this is (its plain form, with no Pictures button
   anywhere, on a sheet whose page copy says pictures_possible false; on a
   sheet of points with places, what to judge from; on a two-map sheet,
   without looking at the two maps, J28), and the one line on Cannot tell
   (or a certificate's own need; none on a two-map sheet); the third, that
   there is no result until every answer is given, is gone (r5-e H1: the
   greyed finish button and the counter say it). Then the face's
   instruction where it has one (right or
   wrong, the class seen with the map's period, a two-map sheet's; a class
   face whose period is not known has its line in the fold, r5-d R08) and the
   row of switches only where one exists (a two-map sheet's blind switch).
   Right above the answers and above the list (J14), the labelling rule of
   the page copy, at body size with "Rule: " in bold (data.reading_rule: the
   owner's, or the one the user typed on the Check card before the draw; on
   a face that takes Cannot tell and can show pictures, the cloud line under
   it, r4-m M8), in one box with a border with that line and the example
   pictures (r5-e H2; who set the rule and when, data.rule_scope, is in the
   fold; since r5-l L1 no line after it says which maps the rule is for), from the
   first open; with no rule on the page copy, on a sheet that can show
   pictures, one line in its place says none was set and what to do (J26),
   and the pane no longer says it (a page copy with no rule and pictures
   with one: the pane shows the rule's two halves and, on a face with Cannot
   tell, the cloud line, r4-m M8, and the face's line goes). ONE closed
   "Details" fold under the answers holds the sheet's file name, a class
   face's line when the map's period is not known (r5-d R08), the rule's
   scope (r5-e H2), why a graded sheet's answers must not change (r5-e H5), a seen
   sheet's first sentence and what Cannot tell does to the result, the keys,
   what never reaches this panel (the blind, code or right-or-wrong line),
   the Street map line, the note that no date is marked clear, that the
   answers are a reading of satellite pictures (J13; r5-s W8), and, on
   a sheet that holds two sets of answers, the server's sentence on them
   (J21). While pictures are off and no pane is open, one line in the pane's
   place, at body size (J05), says how to see them. */

OE.UNJUDGED = '?';
OE.OTHER = '__oe_other_class__';  // the select's 'Other class...' entry

/* The sheet's words from the reader test of 13 Nov 2026 (round 4, pass 2),
   each its FINAL text with window -> the sheet's unit word: the three lines
   on top (what this is, what to do next, what it cannot tell), the note
   after the labelling rule (A/sheet-3), the line while pictures are off
   (A/sheet-1), Cannot tell where it is offered (A/sheet-4), the blind line
   of a seen sheet (A/sheet-5), the graded notice (the owner's extra 2) and
   the saved state (A/sheet-9). None shows a window's class, confidence or
   stratum.

   unit is the page copy's (read_sheet's unit): 'square' where a window is a
   patch of a scores file, 'point' where it is a single pixel at a grid point
   (a plan drawn by result_id); anything else is 'square'. Every word that
   names a window on the face, the list and the pane comes from here (the
   lead's ruling W); tool names, fields and what Claude is told keep
   "window". The lead's first line has a plain form (whatBare) for a sheet
   that can show no pictures (read_sheet's pictures_possible false, round 4
   pass 2, finding 10), and so does Cannot tell (cannotTellBare: no "dates
   shown" where no date is shown); a sheet with places to open says what to
   judge from (whatPlaced, reader test 2's E/sheet-1; squares or points,
   r5-b J11).

   The last round (r4h, reader test 2): one Cannot tell line on the face
   (cannotTellLine, S1), the rest of what Cannot tell does in the Details
   fold (cannotTell, A/F04), the fold's blind and code lines (S1), the
   finality line (everyDone: S2), the class face's period
   (classWhat, E/sheet-3), and "Street map" for the map button (A/F21).

   This round (r4i, 14 Nov 2026): the face's Cannot tell line says what a
   guess costs as well (F5), and the fold's cost sentence says the grid of
   squares stays with Cannot tell and only the per-class lines go (F5, the
   owner's O2).

   Round r5-b (17 Nov 2026, the user-side audit): the lead is three lines
   (what, the Cannot tell line, noResult: J13; next, nextBare, cannot,
   cannotBare and changeLine are gone, and the reading line is in the fold);
   a sheet with no pictures and no places says it cannot show where a point
   is (whatBare, J11); the graded notice in words that are on a button
   (graded, J06); the footer once the sheet changed elsewhere
   (changedElsewhere, J15), or after a removal whose sheet could not be read
   again (notReadAgain); no rule in words that say what to do (noRule, J26;
   noRulePair on a two-map sheet); a two-map sheet's hidden sentence
   (hiddenPair, J28), what its first line says once the maps' answers are
   shown (shownPair) and its no-pictures, no-places line (whatBarePair); and
   the finish line's second sentence with its object (everyDone, J29).

   Round r5-d (17 Nov 2026, the last audit of the whole app): the cloud line
   says a case, not a fact about the square shown (cloudLine, R02); a sheet
   that already gave its result says so beside the finish button too
   (everyDoneGraded, R06); the class face's line with no period known is in
   the Details fold (classWhat, R08); the list's menu says Cannot tell as
   the face's button does (R13).

   Round r5-e (17 Nov 2026, one design style for every view): the lead is
   two short lines (H1): what to do in one clause, no longer "hidden" and
   "do not look" in two (what, whatPlaced; hiddenOne, hiddenPair and
   shownPair are the clauses swapped on a two-map sheet), and the Cannot
   tell line without the reason, which stays in the fold (cannotTellLine);
   noResult is gone, since the greyed finish button and the counter say it;
   the class face's line with the period starts with what to do (classWhat)
   and the pictures line with the press (picturesOff; in the list, a row's
   Show: picturesOffList, H4). The finish line is its second sentence alone
   (everyDone, H5), and the graded notice leaves its reason to the fold
   (graded, gradedWhy).

   Round r5-f (18 Nov 2026, the audit after the design round): while a
   square has no answer the footer says why the finish button is grey
   (notYet, F02: r5-e H1 left that to the button and the counter, which say
   it in no words); and a sheet with no pictures and no places ends its line
   with the way out as a question of its own (whatBare, whatBarePair, F12). */
// WORDING: squares
OE.sheetWords = (unit) => {
  const one = unit === 'point' ? 'point' : 'square';
  const One = one === 'point' ? 'Point' : 'Square';
  const many = one + 's';
  // What Cannot tell does to the result, after its first sentence (A/F04).
  // WORDING: r4i F5 O2
  const cannotTellRest = ' Do not guess: a guess can make the result wrong. A Cannot tell does not: you still'
    + ' get a result, with a wider range, but without the per-class lines.';
  return {
    one,
    many,
    // "Without looking at the map" says whoever answers must not look at the
    // map itself (r4-m M6: a map's maker answering would grade their own
    // work); the same words on every sheet, squares or points. One clause
    // (r5-e H1: "hidden" and "do not look" were two clauses of one meaning).
    // WORDING: r5e H1
    // WORDING: r4m M6
    what: 'Say what you see in each ' + one + '\'s satellite pictures, without looking at the map.',
    // The clause what and whatPlaced end with, and what a two-map sheet's
    // lead says in its place (r5-b J28; r5-e H1: a clause, no longer a
    // sentence of its own).
    // WORDING: r5e H1
    // WORDING: r5b J28
    hiddenOne: ', without looking at the map.',
    // WORDING: r5e H1
    // WORDING: r5b J28
    hiddenPair: ', without looking at the two maps.',
    // In hiddenPair's place while the reviewer has unticked the switch and
    // the face shows what the two maps say (r5-b J28): the sentence ends, and
    // this one says the answers are shown (r5-e H1), and, since the answers
    // lock once they are shown (r5-m B2), that the answers are locked; it is
    // the locked answers' description then (lockAnswers).
    // WORDING: r5m B2
    // WORDING: r5e H1
    // WORDING: r5b J28
    shownPair: '. The two maps\' answers are shown now, so your answers are locked.',
    // In hiddenPair's place once the maps' answers were shown and the switch
    // is ticked again, or locked again by an answer taken in from elsewhere
    // (r5-m B2): the answers stay locked (state.revealed), and the line is
    // still their description (lockAnswers), so it says so; the hidden form
    // asked for answers the lock refuses.
    // WORDING: r5m B2
    seenPair: '. The two maps\' answers were shown, so your answers are locked.',
    // The tail whatBarePair ends with, word for word (it holds no hiddenPair
    // clause): shownPair or seenPair takes its place once the maps' answers
    // were shown (r5-m B2), since the answers are locked then and nothing is
    // left to answer from records.
    barePairTail: ', without looking at the maps. No records? Stop and tell Claude.',
    // In the Details fold of a sheet that can show pictures (r5-b J13;
    // before, on the lead's third line). r5-s W8: it says what the answers
    // are and no more (it ended with the caution on what they are not; that
    // is in the model's line, the server's SAY_LIMITS, and the docs).
    // WORDING: r5s
    // WORDING: r5b J13
    reading: 'Your answers are a reading of satellite pictures.',
    // A sheet with no pictures and no places cannot show where a point is
    // (r5-b J11: "say what is there" left every answer a guess): it says
    // what to answer from, and to stop with nothing to answer from. The way
    // out is a question and an answer of its own (r5-f F12: as the tail of
    // the sentence before it, "none" had nothing clear to point at).
    // WORDING: r5f F12
    // WORDING: r4m M6; r5b J11
    whatBare: 'This sheet has no pictures and no places, so it cannot show where a ' + one + ' is. Answer only from'
      + ' your own records of each ' + one + ', without looking at the map. No records? Stop and tell Claude.',
    // whatBare on a two-map sheet: the maps (r5-b J28; a two-map sheet never
    // takes whatBare, and "the maps." is not the clause its lead swaps).
    // WORDING: r5f F12
    // WORDING: r5b J28
    whatBarePair: 'This sheet has no pictures and no places, so it cannot show where a ' + one + ' is. Answer only'
      + ' from your own records of each ' + one + ', without looking at the maps. No records? Stop and tell Claude.',
    // A sheet with places to open (data.has_place), squares or points (r5-b
    // J11): no pictures, what to judge from; ending with what's own clause
    // (r5-e H1), so a two-map sheet swaps it the same way.
    // WORDING: r5e H1
    // WORDING: r4h S3 E/sheet-1; r4m M6
    whatPlaced: 'This sheet has no pictures. See where each ' + one + ' is, then say what was there from your own'
      + ' imagery or records, without looking at the map.',
    // r5-p: the pane now folds the dates the cloud mask marks, so "the panel
    // marks no date as clear" was no longer true; the mask is a guide only.
    // WORDING: r5p
    clear: 'Sentinel-2\'s own cloud mask is a guide and is sometimes wrong: you decide, from the outlined square'
      + ' itself. Scene cloud is for the whole satellite image, not this square.',
    // Under the rule, on a face that takes Cannot tell and can show pictures
    // (r4-m M8): the sheet's rule block, or the pictures pane where the page
    // copy has no rule. Only the case the rule leaves open: one clear date
    // is enough for Present, and Absent counts the clear dates alone, so a
    // square clouded on every date is Cannot tell, not Absent by default.
    // Worded as a case, not a fact (r5-d R02): it stands above the face and
    // the list before any square's pictures are open, and "No clear date of
    // this square" read as said of the square shown.
    // WORDING: r5d R02
    // WORDING: r4m M8
    cloudLine: 'If cloud covers the ' + one + ' on every date, press Cannot tell.',
    // The pane's closed fold that holds each date's scene cloud, off the
    // dates' headings (r4-m M8), and its first line, clear's own second
    // sentence.
    // WORDING: r4m M8
    cloudFold: 'Scene cloud of each date',
    // WORDING: r4m M8
    cloudWhole: 'Scene cloud is for the whole satellite image, not this ' + one + '.',
    // The pane's one line when no month of the map's period has a scene
    // with data here (OE.NO_IMAGE, r5-a L2), in place of a list of empty
    // months; a face that takes Cannot tell adds OE.NO_IMAGE_PRESS.
    // WORDING: r5a L2
    noImage: OE.NO_IMAGE(one),
    // In the pane's place while pictures are off: the press first (r5-e H1:
    // it opened with "Nothing to look at yet").
    // WORDING: r5e H1
    picturesOff: 'Press Pictures to see satellite pictures; you are asked first what is sent.',
    // The same line in the list, where the press is a row's Show under the
    // head "Pictures" (r5-e H4: no button there reads "Pictures").
    // WORDING: r5e H4
    picturesOffList: 'Press Show under Pictures to see a ' + one + '\'s satellite pictures; you are asked first what is sent.',
    // The face's one Cannot tell line, where the sheet takes Cannot tell; what
    // a guess costs is in the fold (cannotTell; r5-e H1).
    // WORDING: r5e H1
    // WORDING: r4i F5
    cannotTellLine: 'Cannot judge a ' + one + '? Press Cannot tell; do not guess.',
    // In the Details fold (A/F04).
    // WORDING: r4i F5 O2
    cannotTell: 'Cannot tell is for a ' + one + ' you cannot judge from the dates shown.' + cannotTellRest,
    // A sheet that can show no pictures (pictures_possible false) shows no date.
    // WORDING: r4i F5 O2
    cannotTellBare: 'Cannot tell is for a ' + one + ' you cannot judge.' + cannotTellRest,
    // In the Details fold (B/sheet-blind-line).
    // WORDING: r4h S1
    blindSeen: 'Blind: this panel is not sent what the map says for a ' + one + ', or where the ' + one
      + ' is, so you answer without seeing the map\'s answer.',
    // In the Details fold, where the sheet has places (A/F21).
    // WORDING: r4h A/F21
    openMap: 'Street map fetches one ' + one + '\'s place only when you press it; it never shows the map\'s answer.',
    // The map button and the list's column (A/F21).
    // WORDING: r4h A/F21
    streetMap: 'Street map',
    // The notice on a sheet already graded (the owner's extra 2), in words
    // that are on a button (r5-b J06: "graded" was on none); its reason is
    // in the Details fold (gradedWhy, r5-e H5), so the notice is two lines.
    // r5-k K3: the sheet's answers are locked now (lockAnswers), so its first
    // sentence says so, and it is the locked answers' description; it said
    // "Do not change an answer now", which the panel no longer allows. The
    // footer's line beside the finish button keeps its shorter words
    // (everyDoneGraded, r5-d R06), which say the same first fact.
    // WORDING: r5k K3
    // WORDING: r5e H5
    // WORDING: r5b J06
    graded: 'This sheet gave its result; its answers are locked. A second opinion needs another person and a fresh'
      + ' sheet.',
    // r5-k K2: under the words of a two-map sheet's switch while it is
    // locked: it shows what the two maps say only once every square is
    // answered and saved and the sheet's result was asked for (the finish
    // press Claude took).
    // WORDING: r5k K2
    pairLocked: 'Shown once every ' + one + ' is answered and the result is sent.',
    // In pairLocked's place once the finish press's message was shown to copy
    // (the host did not take it), so the result may have been sent by hand:
    // the switch opens once the sheet gave its result, and a sheet that gave
    // its result opened again is unlocked.
    // WORDING: r5k K2
    pairLockedNotSent: 'Shown when you open this sheet again after Claude grades it.',
    // In the Details fold of a sheet already graded: why an answer must not
    // change now (r5-e H5; before, the middle clause of the notice).
    // WORDING: r5e H5
    gradedWhy: 'A change made after seeing where the map disagrees makes the result invalid.',
    // Plain text beside the counter while nothing waits to be saved (B/sheet-finish-button).
    saved: 'Saved automatically',
    // Beside "Reopen the sheet" once the sheet changed under the page (r5-b
    // J15): what happened, whether answers given here are lost (k of them
    // were not saved), and what to do next.
    // WORDING: r5b J15
    changedElsewhere: (k) => (k
      ? 'This sheet was changed somewhere else, so ' + k + (k === 1 ? ' answer here was' : ' answers here were')
        + ' not saved. Reopen the sheet, then answer ' + (k === 1 ? 'that ' + one : 'those ' + many) + ' again.'
      : 'This sheet was changed somewhere else. Reopen the sheet to carry on.'),
    // In changedElsewhere's place when this panel removed a set of answers
    // and then could not read the sheet again (markStale('read')): nothing
    // else changed it, and the toast says which answers went.
    // WORDING: r5b J15
    notReadAgain: 'This panel could not read the sheet again. Reopen the sheet to carry on.',
    // The class face's instruction, with the map's period where the sheet knows it (E/sheet-3); with
    // it, the line starts with what to do, no longer as the lead starts (r5-e H1); without it,
    // unchanged and in the fold (r5-d R08).
    // WORDING: r5e H1 (with the period)
    // WORDING: r4h S3 E/sheet-3
    classWhat: (period) => (period
      ? 'Pick what was there during ' + period + ', using the map\'s class names.'
      : 'For each ' + one + ', pick what was there in the period the map covers, using the map\'s class names;'
        + ' ask whoever made the map for that period.'),
    // WORDING: r5b J26
    noRule: 'No rule was set for this map. Decide your rule before the first answer and use it for every ' + one
      + '.',
    // noRule on a two-map sheet.
    // WORDING: r5b J26 J28
    noRulePair: 'No rule was set for these two maps. Decide your rule before the first answer and use it for every '
      + one + '.',
    // The face, the list and the pane (the lead's ruling W).
    title: (k, n) => One + ' ' + k + ' of ' + n,
    oneView: 'One ' + one,
    howToShow: 'How to show the ' + many,
    prevLabel: 'Previous ' + one,
    nextLabel: 'Next ' + one,
    // r5-r N1: the third button beside Previous and Next while a square has
    // no answer, and its name for a screen reader (never "Next square" in a
    // row: that is Next's own name).
    // WORDING: r5r
    nextOpen: 'Next unlabelled',
    // WORDING: r5r
    nextOpenLabel: 'Next unlabelled ' + one,
    // r5-r N1: the key's sentence, at the end of the face's keys line, where
    // no answer of the sheet has the key U, while a square has no answer.
    // WORDING: r5r
    nextOpenKey: ' U shows the next unlabelled ' + one + '.',
    // r5-r N2: beside the counter while 1 to 3 squares are left: the line's
    // start, then the squares by their place ("Square 29"; "Squares " before
    // "7, 29 and 80"), each a text button named goTo.
    // WORDING: r5r
    leftLead: 'Unlabelled: ',
    // WORDING: r5r
    leftOne: (k) => One + ' ' + k,
    // WORDING: r5r
    leftMany: One + 's ',
    // WORDING: r5r
    goTo: (k) => 'Go to ' + one + ' ' + k,
    none: 'No ' + many,
    pictures: (k) => 'Pictures: ' + one + ' ' + k,
    // A row's pictures button in the list (r5-e H4, after review): its name
    // starts with the word it shows, Show, or Hide on the row whose pictures
    // are open (a press there closes them).
    // WORDING: r5e H4
    showPictures: (k) => 'Show pictures: ' + one + ' ' + k,
    // WORDING: r5e H4
    hidePictures: (k) => 'Hide pictures: ' + one + ' ' + k,
    openRow: (k) => 'Open ' + one + ' ' + k,
    // WORDING: r4h A/F21
    openMapOf: (k) => 'Street map: ' + one + ' ' + k + ' on OpenStreetMap',
    noLink: (k) => 'No map link for ' + one + ' ' + k + '.',
    seenIn: (k) => 'Class seen in ' + one + ' ' + k,
    rightIn: (k) => 'Is the map right in ' + one + ' ' + k,
    otherIn: (k) => 'Other class seen in ' + one + ' ' + k,
    okIn: (k) => 'OK: class seen in ' + one + ' ' + k,
    moreIn: (k) => 'More classes for ' + one + ' ' + k,
    paneHead: (k) => One + ' ' + k,
    // A window's name for a screen reader, in its pictures and dates.
    label: (k) => one + ' ' + k,
    back: (k) => 'Back to ' + one + ' ' + k,
    notOfSheet: 'Not a ' + one + ' of this sheet: nothing here is labelled.',
    // "Square 4", "Squares 4 and 9", "Squares 2, 4 and 9": windows by their place.
    places: (sorted) => (sorted.length === 1 ? One + ' ' + sorted[0]
      : One + 's ' + sorted.slice(0, -1).join(', ') + ' and ' + sorted[sorted.length - 1]),
    rereadMine: (name) => 'Saved to ' + name + '; the sheet was read again for the ' + many
      + ' the server worked out.',
    // Beside the finish button once every answer is saved (A/F17); the
    // second sentence has its object now (r5-b J29); the first, which said
    // again what the counter and "Saved automatically" say, is gone (r5-e H5).
    // WORDING: r5e H5
    // WORDING: r4h S2 A/F17; r5b J29
    everyDone: 'After you press, do not change an answer.',
    // In everyDone's place while a square has no answer yet (r5-f F02): why
    // the finish button is grey, in words (the grey button and the counter
    // say it in none). r5-r N2: it ends with a text button that shows the
    // next square with no answer (goOpen, then a full stop): "Answer every
    // square first: go to the next unlabelled."
    // WORDING: r5r
    // WORDING: r5f F02
    notYet: 'Answer every ' + one + ' first: ',
    // WORDING: r5r
    goOpen: 'go to the next unlabelled',
    // In everyDone's place on a sheet that already gave its result (r5-d
    // R06): its notice says not to change an answer now, so the footer does
    // not say an answer may change until the press (r5-k K3 left it as it
    // was: the notice's longer first sentence would not fit the footer's one
    // row at 780 px beside the finish button, r5-e H5).
    // WORDING: r5d R06
    everyDoneGraded: 'This sheet already gave its result.',
    // The finish button's words on a sheet that already gave its result,
    // once it is ready (r5-m B5, finishLabel): its answers are locked (r5-k
    // K3) and its press gives the same result again, so the button no longer
    // invites a new one (it read the job's own label, "Estimate the error
    // rate"). The message it sends is unchanged; a sheet not graded, and a
    // graded one not ready (grey: a blank square, two sets of answers),
    // keeps its job's label.
    // WORDING: r5m B5
    gradedFinish: 'Show the result again',
    // In gradedFinish's place where the answers changed since that first
    // result (the page copy's graded changed_since_first): the press then
    // gives the withdrawn card, "This result is out of date ... so this
    // result does not count", not the result again (r5-m B5, after review).
    // WORDING: r5m B5
    gradedChanged: 'See why the result does not count',
    // In notYet's place on a sheet that already gave its result while a
    // square has no answer (one blanked elsewhere after the grading): its
    // answers are locked here (r5-k K3), so notYet would ask for an answer
    // the panel no longer takes; this says what is missing and what a new
    // result needs, as the graded notice says of a second opinion.
    // WORDING: r5k K3
    gradedBlank: 'A ' + one + ' has no answer; a new result needs a fresh sheet.',
    // In the Details fold, where the sheet has a blind switch (C/sheet-3).
    // WORDING: r4h S1 C/sheet-3
    codeLine: 'While you label, this panel is not sent a ' + one + '\'s number, place or order.',
    wrongLine: 'Right or wrong shows the map\'s class; the ' + one + '\'s index, place, order and stratum are'
      + ' not sent.',
    listAfter: ' After an answer the next ' + one + ' gets the focus.',
    faceKeys: ' left and right arrow keys move between ' + many + '. After an answer the next ' + one
      + ' not yet labelled shows.',
  };
};

/* The words of the server's refusal of a save from a page drawn before the
   sheet changed (sheet.CHANGED_UNDER_PAGE, which a test holds this to): a
   retry fails the same way, so the page offers to reopen the sheet. */
OE.CHANGED_UNDER_PAGE = 'This sheet changed since the panel opened it';
/* The server's refusals of a class map's other face (sheet.FACE_WRONG_REFUSED
   and sheet.FACE_CLASS_REFUSED, up to their ';', which a test holds these to). */
OE.FACE_WRONG_REFUSED = "Right or wrong shows the map's class, so it is offered only before any class seen is saved";
OE.FACE_CLASS_REFUSED = 'This sheet holds right or wrong labels';

/* How this page shows a sheet's windows, as the user last chose it: 'face'
   (one window at a time, the default) or 'list'. An editor drawn in place
   of another (a face switched, a sheet reopened) keeps it. */
OE.sheetView = 'face';

/* The host's display mode changed: every label editor on the page lays its
   face out for it (the dates a grid in full screen, a strip inline). A
   context change that names no display mode changes nothing. OE.displayMode
   is common.js's now (r4-k K5: it follows the host's context and the Full
   screen button's answers); this keeps it as it was given. */
OE.sheetMode = (mode) => {
  if (typeof mode !== 'string' || !mode) return;
  OE.displayMode = mode;
  for (const editor of document.querySelectorAll('section.sheet')) {
    if (typeof editor.oeMode === 'function') editor.oeMode(mode);
  }
};

/* Every label editor inside node is about to be replaced (a page drawn
   again, another zone's sheet opened): each drops its pictures timer and its
   page-wide listeners (its oeDispose). */
OE.disposeEditors = (node) => {
  for (const editor of node ? node.querySelectorAll('section.sheet') : []) {
    if (typeof editor.oeDispose === 'function') editor.oeDispose();
  }
};

/* Two example pictures beside the owner's rule (r4-k K9): the page copy's
   examples, { present, absent, year }, which the server sends only with
   the owner's rule and both pictures and their notes on disk; null when
   the rule is not the owner's or anything is missing (each picture a
   data URL, the year a whole number). A public place, never the map being
   checked, so a blind sheet stays blind. */
OE.ruleExamples = (data) => {
  const ex = data && data.rule_source === 'owner' && data.examples && typeof data.examples === 'object'
    ? data.examples : null;
  const good = (src) => typeof src === 'string' && src.startsWith('data:image/');
  if (!ex || !good(ex.present) || !good(ex.absent) || !Number.isInteger(ex.year)) return null;
  return { present: ex.present, absent: ex.absent, year: ex.year };
};
/* The pictures' captions, and the line in the Details fold that says
   where they come from (r4-k K9). r5-l L2 (the owner, 19 Nov 2026: the
   instructions are general): each caption says what Present or Absent means
   in general words, never "under this rule" of one kind of map; one small
   line under the pictures says where they come from (OE.EXAMPLES_SOURCE, the
   only words of the sheet's own that name kelp: the pictures' source, never
   the question asked). The Check card's words, word for word. r5-m B1: each
   says what the map is meant to find, as the app's rule does (r5-l said
   "what the map shows", read as the map's own answer, which a blind sheet
   hides). */
// WORDING: r5m B1
// WORDING: r5l L2
// WORDING: r4k K9
OE.EXAMPLE_PRESENT = 'Present: what the map is meant to find covers more than half of the square on a clear date.';
// WORDING: r5m B1
// WORDING: r5l L2
// WORDING: r4k K9
OE.EXAMPLE_ABSENT = 'Absent: what the map is meant to find covers half of the square or less on every clear date.';
// WORDING: r4k K9
OE.EXAMPLES_NOTE = (year) => 'A public place, not your map. Contains modified Copernicus Sentinel data ' + year + '.';
// WORDING: r5l L2
OE.EXAMPLES_SOURCE = 'Example pictures from a kelp canopy map.';
/* The two pictures side by side, small, each with its caption (also its
   alt text; the caption itself is not read a second time). 120 px wide, as
   the Check card's own pair (check.html .oe-examples), so on that card the
   editor's pair fits the same figure box; r5-l L2: each figure 120 px wide
   too, so a caption of a sentence wraps under its picture, not wider. Under
   the pair, in the same block (class oe-examples-set, as on the Check card),
   the one small line on where they come from, in grey as on the Check card,
   4 px under the captions. */
OE.examplePictures = (ex) => OE.el('div', { class: 'oe-examples-set' },
  OE.el('div', { class: 'row oe-examples', style: 'gap:12px;align-items:flex-start' },
    [[ex.present, OE.EXAMPLE_PRESENT], [ex.absent, OE.EXAMPLE_ABSENT]].map(([src, caption]) => OE.el('figure', { style: 'margin:0;width:120px;max-width:100%' },
      OE.el('img', { src, alt: caption, style: 'display:block;width:120px;max-width:100%;height:auto;border:1px solid var(--rule);border-radius:4px' }),
      OE.el('figcaption', { class: 'small', 'aria-hidden': 'true' }, caption)))),
  OE.el('div', { class: 'small muted oe-examples-from', style: 'margin-top:4px' }, OE.EXAMPLES_SOURCE));

/* The pictures' parts, shared by the label editor's pane and a review set's
   window pane (result.html): pure functions of what they are given, so the
   editor's pictures are drawn as before. */

/* How long the page waits for a picture call (r5-a L1): 200 s. The server
   can take about 160 s on one call (up to 40 s for its turn, 40 s for the
   sheet's search by another square, then its own search and its reads, 40 s
   each), and the client's own default is shorter than that. Passed as the
   call's timeout (OE.call's options) by the label editor's pane, its test
   scene, and a review set's pane and test scene (result.html). */
OE.PICTURES_WAIT_MS = 200000;
/* A picture call that threw: it got no answer (the host's timeout, a dropped
   bridge), or a failure the server did not put in one of its own sentences.
   The pane then says this one fixed sentence, never the thrown error's own
   text, with Try again (r5-a L1). */
// WORDING: r5a L1
OE.PICTURES_NOT_ARRIVED = 'The pictures did not arrive. Press Try again.';
/* The pane's button under such a failure, and under the server's own
   sentences another try can get past (its answer's again: a slow fetch, the
   catalogue, the image files): the same call once more, for the same square. */
// WORDING: r5a L1
OE.TRY_AGAIN = 'Try again';
/* What a pane shows for a picture call that threw: the fixed sentence, with
   again. The error itself is not passed in, so its text cannot be shown. */
OE.picturesFailed = () => ({ ok: false, error: OE.PICTURES_NOT_ARRIVED, again: true });
/* No month of the map's period has a scene with data on this square (the
   server's no_image, r5-a L2): one fixed line in the pane, in place of a
   list of empty months, in the pane's unit word; then, on a face that takes
   Cannot tell, what to press. A search the catalogue cut short is told by
   the record beside the sheet, never here. */
// WORDING: r5a L2
OE.NO_IMAGE = (unit) => 'No satellite image was found for this ' + (unit === 'point' ? 'point' : 'square')
  + ' in the map\'s period.';
// WORDING: r5a L2
OE.NO_IMAGE_PRESS = ' Press Cannot tell.';

/* The question before pictures (r5-k K7, the owner's choice N1 of 18 Nov
   2026), drawn alike by the label editor's pane and a review set's pane
   (result.html). Its face: the server's three short lines (its ask,
   imagery.CONSENT_ASK, one line each between its line breaks: what is sent,
   to whom, and that nothing goes to Studio or Claude), then "Turn on
   pictures", then one closed fold, OE.ASK_FOLD, then "Try the public test
   scene" (each pane draws the two buttons itself). The fold stands where
   the test scene's grey paragraph stood: it holds the two hosts by name and
   what each gets (ask_full, imagery.CONSENT_FULL), then the test scene's
   question (test_ask), word for word, at the small size in the text's
   colour (on the face they were about 11 lines, then a grey paragraph), so
   the test scene's button comes after the fold that says what it sends. A
   fold with nothing to hold is not drawn. */
// WORDING: r5k K7
OE.ASK_FOLD = 'What is sent, in full';
OE.askLines = (answer) => OE.el('div', { class: 'oe-ask-lines' },
  String((answer && answer.ask) || '').split('\n').filter((line) => line.trim())
    .map((line, i) => OE.el('div', { class: 'oe-ask-line', style: i ? 'margin-top:4px' : null }, line)));
OE.askFold = (answer) => {
  const said = [answer && answer.ask_full, answer && answer.test_ask]
    .filter((s) => typeof s === 'string' && s.trim());
  return said.length
    ? OE.el('details', { class: 'oe-ask-full' }, OE.el('summary', {}, OE.ASK_FOLD),
      said.map((s) => OE.el('div', { class: 'small', style: 'margin-top:8px' }, s)))
    : null;
};

/* A picture the server sent: a data URL and whole-number sizes. */
OE.goodPicture = (p) => Boolean(p && typeof p.src === 'string' && p.src.startsWith('data:image/')
  && Number.isFinite(Number(p.width)) && Number.isFinite(Number(p.height)));

/* One picture, the outlined square drawn over it (never into its pixels),
   with its caption ("Context, square outlined", "Context, point outlined"
   by unit, or "Detail", unless caption is given: the owner's ruling 6, no
   "window" on a face). date and what name it for a screen reader. The
   outline is 2 px of yellow with a 1 px black edge outside and inside it
   (r5-j B2: the yellow alone was lost on cloud, sand or snow); the edge is
   a box-shadow, and the box sits 3 px outside the square (it sat 2 px out),
   so the inner black line lies between the yellow and the square and covers
   none of the square's own pixels. */
OE.picture = (p, date, what, caption = null, unit = 'square') => {
  const s = Number(p.scale) || 1;
  const b = p.box && typeof p.box === 'object' ? p.box : null;
  const w = Number(p.width), h = Number(p.height);
  const outline = b && [b.x, b.y, b.w, b.h].every((v) => Number.isFinite(Number(v)))
    ? OE.el('div', { 'aria-hidden': 'true', style: 'position:absolute;box-sizing:border-box;pointer-events:none;border:2px solid #ffd600;box-shadow:0 0 0 1px #000,inset 0 0 0 1px #000;'
        + 'left:' + (Number(b.x) * s - 3) + 'px;top:' + (Number(b.y) * s - 3) + 'px;width:' + (Number(b.w) * s + 6) + 'px;height:' + (Number(b.h) * s + 6) + 'px' })
    : null;
  return OE.el('figure', { style: 'margin:0' },
    OE.el('div', { style: 'position:relative;outline:1px solid var(--rule);border-radius:4px;width:' + w + 'px;height:' + h + 'px' },
      OE.el('img', { src: p.src, width: w, height: h, alt: String(p.label || '') + ', ' + date + ', ' + what, style: 'image-rendering:pixelated;display:block' }),
      outline),
    // WORDING: squares
    OE.el('figcaption', { class: 'small' }, caption || (p.view === 'detail' ? 'Detail'
      : 'Context, ' + (unit === 'point' ? 'point' : 'square') + ' outlined')));
};

/* One block per date (a month with no scene says so), the chosen rendering
   only (colour: 'nir' or 'true'), with the switch between near-infrared and
   true colour, whose press calls choose(kind). strip: the dates side by
   side in a named region (the face's and the look-first pane's), else one
   under another. r5-k K4 (the owner's O8): the strip wraps its dates onto
   as many rows as it needs, each date at its own size (common.js
   .face-dates); it scrolled sideways, cut at the panel's edge, and took the
   focus itself so the keyboard could scroll it, which it no longer needs
   (no tabindex: a stop that scrolls nothing). unit: the unit word the
   pictures' captions name the outlined window by ('square' or 'point').
   The switch is the joined toggle the sheet's views use, the chosen one
   filled (reader test 2's A/F23), each button keeping its data-focus key;
   each date is its own outlined block, its label on top inside the outline
   and its pictures under it (A/F24). heading: what a date's block is headed
   by, from its label (the label sheet's pane leaves its scene cloud out,
   r4-m M8); by default the label as it is. The pictures' names keep the
   whole label.
   r5-p (the owner, labelling live: "a lot of can't-decide cloud pictures"):
   a date whose scene's own cloud mask marks cloud or shadow on the outlined
   square (the server's cloud_here: 'covered', imagery._dates) is folded
   away, in one closed fold right after the shown dates (.oe-cloudy-dates),
   whose summary says how many (OE.CLOUDY_DATES) and whose first line says
   whose mark it is and that it is sometimes wrong (OE.CLOUDY_WHY); the
   folded dates keep their pictures and their month order. When every date
   with a scene is marked, nothing is folded: all the dates show and one
   line above them says so (OE.ALL_CLOUDY, .oe-all-cloudy). A date with no
   mask ('unknown', or no cloud_here at all: an older server) is shown as
   before, never folded. The face, the list, the test scene and a review
   set's pane (result.html) all draw their dates here, so all fold alike.
   The fold stays open across a redraw of the same pictures (the colour
   switch, an answer): OE.cloudyOpen keeps the names (what) whose fold the
   user opened. Each date's block carries data-cloud (its cloud_here). */
// WORDING: r5p
OE.CLOUDY_DATES = (n) => (n === 1 ? '1 cloudy date' : n + ' cloudy dates');
// WORDING: r5p
OE.CLOUDY_WHY = (unit) => 'Sentinel-2\'s own cloud mask marks cloud or shadow on the ' + (unit === 'point' ? 'point' : 'square')
  + ' on these dates, or it is white in the picture. Both are sometimes wrong (bright roofs, sand, surf, snow).';
// WORDING: r5p
OE.ALL_CLOUDY = (unit) => 'Sentinel-2\'s own cloud mask marks cloud or shadow on the ' + (unit === 'point' ? 'point' : 'square')
  + ' on every date, or it is white in the picture.';
OE.cloudyOpen = new Set();
OE.datesBlock = (dates, what, { colour = 'nir', choose = null, strip = false, heading = null, unit = 'square' } = {}) => {
  // "False colour" names the near-infrared rendering (r5-b J24: the longer
  // name wrapped the switch to two lines on a narrow panel; the caption
  // under the pictures says what the colours show).
  // WORDING: r5b J24
  const kinds = [['nir', 'False colour'], ['true', 'True colour']];
  const colours = OE.toggle(kinds, colour, (kind) => { if (choose) choose(kind); },
    { label: 'Colours of the pictures', small: true });
  colours.querySelectorAll('button').forEach((b, i) => b.setAttribute('data-focus', 'colour:' + kinds[i][0]));
  const all = (Array.isArray(dates) ? dates : []).filter((d) => d && typeof d === 'object');
  const one = (d) => {
    const shown = (Array.isArray(d.pictures) ? d.pictures : []).filter((p) => OE.goodPicture(p) && p.kind === colour);
    const label = String(d.label || d.month || '');
    const block = OE.el('div', { class: 'stack', style: 'gap:4px;border:1px solid var(--rule);border-radius:4px;padding:8px' },
      OE.el('div', { class: 'small', style: 'font-weight:600' }, heading ? heading(label) : label),
      shown.length ? OE.el('div', { class: 'row', style: 'align-items:flex-start;gap:12px;flex-wrap:wrap' },
        shown.map((p) => OE.picture(p, String(d.label || ''), what, null, unit))) : null);
    if (typeof d.cloud_here === 'string') block.setAttribute('data-cloud', d.cloud_here);
    return block;
  };
  // r5-p: the dates the cloud mask marks on the square, folded away unless
  // they are every date with a scene (then all show, under one line).
  const cloudy = all.filter((d) => d.cloud_here === 'covered');
  const masked = all.filter((d) => typeof d.cloud_here === 'string');
  const every = cloudy.length > 0 && cloudy.length === masked.length;
  const folds = cloudy.length > 0 && !every;
  const open = folds ? all.filter((d) => d.cloud_here !== 'covered') : all;
  let fold = null;
  if (folds) {
    const key = String(what);
    fold = OE.el('details', { class: 'oe-cloudy-dates', open: OE.cloudyOpen.has(key) },
      OE.el('summary', { class: 'small' }, OE.CLOUDY_DATES(cloudy.length)),
      OE.el('div', { class: 'small', style: 'margin:6px 0 8px' }, OE.CLOUDY_WHY(unit)),
      OE.el('div', strip ? { class: 'face-dates' } : { class: 'stack' }, cloudy.map(one)));
    fold.addEventListener('toggle', () => {
      if (fold.open) OE.cloudyOpen.add(key); else OE.cloudyOpen.delete(key);
    });
  }
  return OE.el('div', { class: 'stack', style: 'gap:10px' },
    OE.el('div', {}, colours),
    every ? OE.el('div', { class: 'small oe-all-cloudy' }, OE.ALL_CLOUDY(unit)) : null,
    OE.el('div', strip
      ? { class: 'face-dates', role: 'region', 'aria-label': 'Dates, ' + what }
      : { class: 'stack' }, open.map(one)),
    fold);
};

OE.sheetEditor = (app, data, { headerAction = null, onSaved = null } = {}) => {
  const face = typeof data.face === 'string' ? data.face : 'wrong';
  const twoMap = face === 'pair';
  const seen = face === 'seen';
  const wrongFace = face === 'wrong';
  // A face whose answers are listed buttons (seen, right or wrong); the others pick from a list.
  const closed = seen || wrongFace;
  // The sheet's words, with its unit word ("square", or "point" for a plan
  // drawn by result_id: the page copy's unit, the lead's ruling W).
  const W = OE.sheetWords(data.unit);
  // Whether this sheet can show pictures at all (the page copy's
  // pictures_possible: its plan has a scores file to place a window by).
  // When it cannot, the lead's first line takes its plain form, the line about
  // pictures being off is left out and no Pictures button is drawn, so the
  // consent question is never asked for nothing (round 4 pass 2, finding
  // 10). A page copy without the field (an older server) shows them, as
  // before.
  const picturesPossible = data.pictures_possible !== false;
  const rule = data.label_rule || null;  // 'seen', 'class', 'wrong' or null (no design file)
  const windows = (Array.isArray(data.windows) ? data.windows : [])
    .filter((w) => w && typeof w.handle === 'string' && w.handle);
  const n = windows.length;
  const byHandle = new Map(windows.map((w) => [w.handle, w]));
  // A window's place in this list, 1 to n: how rows and labels name it.
  const placeOf = new Map(windows.map((w, i) => [w.handle, i + 1]));
  const place = (h) => placeOf.get(h) || 0;
  // The face's answers, each with the keys the server gives it.
  const answers = (Array.isArray(data.answers) ? data.answers : [])
    .filter((a) => a && typeof a.value === 'string' && typeof a.label === 'string')
    .map((a) => ({ value: a.value, label: a.label, keys: (Array.isArray(a.keys) ? a.keys : []).map(String) }));
  const takesQ = answers.some((a) => a.value === OE.UNJUDGED);
  // The classes of a face that picks from a list, in the server's order.
  const listed = answers.filter((a) => a.value !== OE.UNJUDGED).map((a) => a.value);
  // What a window shows beside its answer: the map's class on the right-or-wrong
  // face; on a two-map sheet, at the reviewer's press (unblind), the two maps'
  // classes. Never a stratum (the owner's O1, the last round's F11).
  const extraOf = (w) => ({
    mapClass: w.map_class === undefined ? null : String(w.map_class ?? ''),
    pair: Array.isArray(w.pair) ? w.pair.map((v) => String(v ?? '').trim()) : null,
  });
  const state = {
    view: OE.sheetView === 'list' ? 'list' : 'face',  // one window at a time, or the list
    at: null,  // the window the face shows (its handle)
    blind: true,
    // The extras of an unblinded read are kept once given: ticking blind again hides them.
    revealed: false,
    // The last save the server did not take (not one refused as stale, which
    // is sent again): Save labels stays in view until a save goes through.
    saveFailed: false,
    revealing: false,
    extra: new Map(windows.map((w) => [w.handle, extraOf(w)])),
    labels: new Map(windows.map((w) => [w.handle, String(w.answer ?? '').trim()])),
    // Each window's answer as the server last held it (read or saved): a
    // change kept when the sheet was saved elsewhere is one made on this.
    base: new Map(windows.map((w) => [w.handle, String(w.answer ?? '').trim()])),
    dirty: new Set(),  // handles whose answer changed and is not yet saved
    saving: false,
    // The sheet is being read again to take in a save made elsewhere: saves wait.
    syncing: false,
    other: null,  // the window whose class is being typed in
    filter: 'any',  // which rows the list shows: any, unlabelled or labelled
    images: new Map(),  // handle -> the server's answer (pictures or the question)
    imagesRun: 0,  // bumped when pictures are switched on or off: older answers are dropped
    // The last picture call that failed, while its window's pane stays shown:
    // { h, answer } (r5-a L1), so its sentence and Try again stay when the
    // pane is drawn again. Never in images: the next ask is a new call.
    failed: null,
    shown: null,  // the window whose pictures the pane shows
    test: null,  // the public test scene, when open: { from: handle, answer }
    colour: 'nir',  // which rendering the dates show: nir (false colour) or true
    // Pictures on, as far as the page knows: set by the switch and by the
    // server's answers (pictures, or the question). The pane then follows the focus.
    picturesOn: false,
    // The control the last answer was given from, when no window was left:
    // the finish button takes the focus from it once it shows.
    finishFrom: null,
    // The server refused a save because the sheet changed since this page
    // drew it: nothing more is saved; the page offers to reopen it.
    stale: false,
    // The last toast about that (reopenToast), which oeLeave clears (r4-l L3).
    reopenSaid: '',
    // The server's word on the two label columns: { problem, column } or null.
    mix: data.mix_problem ? { problem: String(data.mix_problem), column: String(data.clear_column || '') } : null,
    clearing: false,
  };
  let lastSelectKey = null;  // the last key on a list: { focus, browse }; browsing does not move on
  let faceListKey = false;  // the face's list of more classes was browsed with a key: no move on
  let picturesTimer = null;  // the pictures following the focus, 300 ms after an answer
  const linking = new Set();  // windows whose map link is being asked for
  // Re-sends in a row after a save refused as stale (at most STALE_RESENDS).
  let staleResends = 0;
  const STALE_RESENDS = 2;
  // A quiet save is under way (oeFlush, when Claude closes the card: r4-k
  // K1): the footer moves no focus meanwhile.
  let quietSave = false;
  // When the version was last asked for or read (check(): at most once every 3 s).
  let lastCheck = Date.now();
  const CHECK_EVERY_MS = 3000;
  // What the last take-in said about windows whose label saved elsewhere won,
  // carried into the next save's words ("Saved to ...", beside the counter,
  // r5-k K5), which clear the counter's line that said it first, so it is
  // not lost under "Saved".
  let keptNote = '';
  const wrap = OE.el('section', { class: 'card sheet' });
  // Studio's counter: "12/200" in 600 20px, then the word, then what is not saved yet.
  const counterNum = OE.el('span', { style: 'font-size:20px;line-height:28px;font-weight:600' });
  const counterNote = OE.el('span', { class: 'small muted' });
  // "Saved automatically" as plain text beside the counter while nothing
  // waits to be saved (reader test 2's B/sheet-finish-button): no button.
  const savedNote = OE.el('span', { class: 'small muted' }, W.saved);
  // r5-k K5 (the owner's O11): each status of this sheet's saves sits by what
  // it is about, no longer in the status line at the page's foot. savedNote,
  // a polite status: a save that went through ("Saved to ...", counterSay)
  // in the place of "Saved automatically", shown only while nothing waits
  // (updateProgress), so it never stands under Save labels or the finish
  // button, nor beside "1 change not saved yet". countSaid, beside the
  // counter: why the counter changed, a save made elsewhere taken in ("This
  // sheet was saved elsewhere; it shows those labels now.", 3g). saveSaid, a
  // line of its own in the footer under the row of the finish button and
  // Save labels, at its right end, for what Save labels causes while it is
  // shown: a save that failed ("Not saved: ...", 3f; Save labels stays shown
  // then) and its press with nothing waiting ("Every change is saved to
  // ..."); a change that waits or a save under way empties a line that is no
  // error (updateProgress). Each is a status in place before its first words
  // (saveSaid and countSaid empty, drawn but out of the flow: common.js),
  // polite, or an alert for an error, as the status line's are (footerSay).
  savedNote.setAttribute('role', 'status');
  savedNote.setAttribute('aria-live', 'polite');
  const countSaid = OE.el('span', { class: 'small muted oe-count-said', role: 'status', 'aria-live': 'polite' });
  const saveSaid = OE.el('div', { class: 'small muted oe-save-said', role: 'status', 'aria-live': 'polite' });
  const progress = OE.el('span', { class: 'row', style: 'gap:6px;align-items:baseline' },
    counterNum, OE.el('span', {}, 'labelled'), counterNote, savedNote, countSaid);
  // r5-r N2: the last of the counter's row while 1 to 3 squares are left and
  // at least one is answered: which squares they are, by their place, each a
  // text button that shows it (updateOpen). No status: it changes with each
  // answer near the end, and the counter beside it is none either.
  const leftLine = OE.el('span', { class: 'small oe-left' });
  // A text button inside a small line (the squares named there, and the jump
  // at the end of the grey finish button's line): underlined, in the line's
  // size, 24 px tall and at least 24 px wide (a square's number may be one
  // digit).
  const LINE_LINK = 'font-size:inherit;line-height:inherit;padding:3px 4px;min-width:24px;'
    + 'text-decoration:underline;text-underline-offset:2px';
  leftLine.hidden = true;
  progress.append(leftLine);
  // A press of Save starts the re-sends after a stale refusal afresh (save()).
  // Shown only while a change waits, a save runs or one failed (updateProgress).
  const saveButton = OE.el('button', { class: 'primary', onclick: () => { staleResends = 0; save(); } }, 'Save labels');
  // Shown only once a save was refused because the sheet changed under the page.
  const reopenButton = OE.el('button', { class: 'primary', onclick: () => reopen() }, 'Reopen the sheet');
  // Beside it, what happened and what to do, at body size (r5-b J15): its
  // words are W.changedElsewhere, set in updateProgress. It is the button's
  // description (r5-i B7), so a screen reader that reaches the button, or is
  // handed the focus there, hears what failed and what to do.
  const staleLine = OE.el('span', { class: 'error', id: 'oe-stale-' + Math.random().toString(36).slice(2) });
  reopenButton.setAttribute('aria-describedby', staleLine.id);
  // Where the finish press says what became of its message (OE.sendJob, r5-b J20): the sent line, or the text to type.
  const finishType = OE.typeSlot();
  const notice = OE.el('div', { class: 'small' });
  // The server's own sentence on a sheet that holds two sets of answers, in
  // the Details fold (r5-b J21); the notice above the answers says it plainly
  // (updateProgress).
  const mixNote = OE.el('div', { class: 'small' });
  mixNote.hidden = true;
  const body = OE.el('tbody');
  const table = OE.el('table', {}, OE.el('thead'), body);
  const done = OE.el('div', { class: 'row' });
  // The instruction on the face (right or wrong, the class seen with the
  // map's period, a two-map sheet's) and, in the Details fold, a seen
  // sheet's first sentence and what Cannot tell does (S1).
  const faceInstruction = OE.el('div', {});
  const instruction = OE.el('div', { class: 'small' });
  // The keys of the answers, small and faint, in the Details fold.
  const hint = OE.el('div', { class: 'small faint' });
  // The pictures of the window the reviewer asked for (Pictures in its row, or on the face).
  const previewId = 'oe-pictures-' + Math.random().toString(36).slice(2);
  const preview = OE.el('div', { class: 'stack', id: previewId, 'aria-live': 'polite' });
  // The labelling rule of the page copy (the owner's, or one the user typed
  // before the draw): shown right above the face from the first open
  // (ruleBlock, A/sheet-2; the face holds the square's heading, its pictures,
  // then its answers since r5-k K4), so the pane does not repeat it.
  const pageRule = typeof data.reading_rule === 'string' && data.reading_rule ? data.reading_rule : '';
  // In the pane's place while pictures are off and no pane is open (A/sheet-1);
  // never on a sheet that can show no pictures. At body size and colour
  // (r5-b J05: small and grey, it was passed over for the answers).
  const picturesOff = picturesPossible ? OE.el('div', {}, W.picturesOff) : null;
  // The list: the table and the pane, beside each other while the pane is open, where there is room.
  const work = OE.el('div', { class: 'sheet-work' },
    OE.el('div', { class: 'table-frame', style: 'max-height:420px' }, table), preview);

  // The face: one window at a time. tabindex -1: the focus waits here after an answer.
  const faceEl = OE.el('section', { class: 'face stack', tabindex: '-1', 'data-focus': 'face', 'aria-label': W.oneView });
  const faceTitle = OE.el('h2', {});
  const faceLive = OE.el('div', { 'aria-live': 'polite' }, faceTitle);
  // Previous, Next, Pictures and Street map are outlined small buttons, as
  // the look-first pane's Previous and Next (r5-b J05: as bare words they
  // did not read as buttons, and a square could be answered with nothing
  // looked at); the list's row buttons stay text buttons.
  const prevButton = OE.el('button', {
    class: 'small', 'data-focus': 'prev', 'aria-label': W.prevLabel, onclick: () => pageBy(-1),
  }, 'Previous');
  const nextButton = OE.el('button', {
    class: 'small', 'data-focus': 'next', 'aria-label': W.nextLabel, onclick: () => pageBy(1),
  }, 'Next');
  // r5-r N1 (the owner, 20 Nov 2026): a third button of the same kind, shown
  // only while a square has no answer (updateOpen): the next square with no
  // answer after the one shown, wrapping round (jumpOpen; true: a press
  // inside the face). Its key is U where no answer of the sheet has that key
  // (a class named "Urban" keeps it: openKey is then null, and the button
  // names no key). Grey (aria-disabled, as Previous at the first square)
  // while the square shown is the only one with no answer (updateOpen).
  const openKey = answers.some((a) => a.keys.includes('U')) ? null : 'U';
  const nextOpenButton = OE.el('button', {
    class: 'small', 'data-focus': 'next-open', 'aria-label': W.nextOpenLabel, 'aria-keyshortcuts': openKey,
    onclick: () => jumpOpen(true),
  }, W.nextOpen);
  nextOpenButton.hidden = true;
  // The row of the three (two once every square is answered): one layout in
  // both states, so Previous and Next keep their size and place when the
  // third button comes or goes. Equal columns at least 132 px wide that
  // wrap, two to a row on a narrow panel, each as wide and as tall as the
  // others: design rule, same kind, same size, at 780 and 380 px. The row
  // flow is set here because common.js gives .face-top > .row a column flow
  // (one row, a new column per button), under which the third button never
  // wrapped; auto-fill keeps the third column's place while its button is
  // hidden.
  const PAGER = 'display:grid;grid-auto-flow:row;grid-template-columns:repeat(auto-fill,minmax(132px,1fr));'
    + 'grid-auto-rows:1fr;align-items:stretch;gap:8px;flex:0 1 412px;min-width:0';
  const pager = OE.el('div', { class: 'row', style: PAGER }, prevButton, nextButton, nextOpenButton);
  const faceExtra = OE.el('div', {});
  const faceAnswerArea = OE.el('div', { class: 'stack' });
  // No Pictures button on a sheet that can show no pictures (finding 10).
  const facePicturesButton = picturesPossible ? OE.el('button', {
    class: 'small', 'aria-controls': previewId, onclick: () => { if (state.at) togglePictures(state.at); },
  }, 'Pictures') : null;
  const faceMapButton = data.has_place ? OE.el('button', {
    class: 'small', onclick: () => { if (state.at) openMap(state.at); },
  }, W.streetMap) : null;
  const facePictures = OE.el('div', {});
  if (n) {
    // r5-k K4 (the owner's O8): the square's pictures come before its
    // answers: the heading with Previous and Next, then Pictures and Street
    // map with the pictures pane (or the line in its place while pictures are
    // off), then what the square shows beside its answer (the map's class on
    // right or wrong, a two-map sheet's pair) and the answers with Cannot
    // tell. The answers stood above the pictures, so a square could be
    // answered before its pictures were looked at.
    faceEl.append(
      OE.el('div', { class: 'face-top' }, faceLive, pager),
      OE.el('div', { class: 'row', style: 'gap:8px' }, facePicturesButton, faceMapButton),
      facePictures,
      faceExtra, faceAnswerArea);
  } else {
    faceEl.append(OE.el('div', { class: 'faint' }, W.none));
  }
  // Holds the face or the list, whichever the user chose.
  const area = OE.el('div', {});

  // A certificate plan (seed_from_map) takes no Cannot tell: its certificate needs 1 or 0.
  const takesUnjudged = data.takes_unjudged !== false;
  // An answer's keys for aria-keyshortcuts ("P 1"), or null when it has none.
  const keysOf = (list, value) => ((list.find((a) => a.value === value) || { keys: [] }).keys.join(' ') || null);
  // An answer counts where it is one the face takes: '?' only where the face lists it.
  const answered = (value) => value !== '' && (value !== OE.UNJUDGED || takesQ);
  const count = () => [...state.labels.values()].filter(answered).length;
  const pending = () => state.dirty.size;
  // An error's words without their last full stop.
  const bare = (text) => String(text || '').replace(/\.+$/, '');
  // One of a sheet's two sets of answers, by what it holds, for the notice
  // of a mix and the toasts of its removal (r5-b J21): the wrong column is
  // "the right-or-wrong marks", the class column "the classes picked";
  // never a column's name ("the wrong column" read as "the incorrect one").
  // WORDING: r5b J21
  const answersOf = (c) => (c === 'wrong' ? 'right-or-wrong marks' : 'classes picked');
  const wait = (ms) => new Promise((resolve) => { setTimeout(resolve, ms); });

  /* Whether the sheet, as the panel holds it, is one the estimate grades. */
  function complete() {
    return n > 0 && count() === n && !state.mix;
  }

  /* What the editor tells Claude (OE.tellClaude, its 'sheet' part): the
     sheet's name and paths and how far it is labelled, k as the counter
     counts it, with how many say Cannot tell; whether every change is saved;
     once complete, the finish button's words, as the button says them (r5-m
     B5: finishLabel, ready then, so W.gradedFinish or W.gradedChanged on a
     sheet that already gave its result). Never a window, a place, a class, a
     label or a count of any other answer. */
  function contextText() {
    const unjudged = takesQ ? [...state.labels.values()].filter((v) => v === OE.UNJUDGED).length : 0;
    const left = pending();
    const design = data.design_path ? ', design_path ' + data.design_path : '';
    let text = 'Label sheet ' + data.name + ' (labels_csv_path ' + data.path + design + '): '
      + count() + ' of ' + n + ' windows labelled' + (unjudged ? ', ' + unjudged + ' cannot tell' : '') + ', '
      + (left ? OE.counted(left, 'change') + ' not saved yet' : 'all saved') + '.';
    if (state.stale) text += ' This panel stopped saving: the sheet changed since it opened it.';
    else if (complete() && !left) {
      text += ' Every window is labelled and saved; its finish button reads "' + finishLabel(true) + '".';
    }
    return text;
  }

  /* Tell Claude how far the sheet is labelled, while this editor is on the
     page: once it is drawn, after each save, after a save made elsewhere is
     taken in, and when it stops saving; never on each answer. */
  function tellClaude() {
    if (wrap.isConnected) OE.tellClaude(app, contextText(), 'sheet');
  }

  /* "a, b or c" */
  const orList = (items) => (items.length > 1
    ? items.slice(0, -1).join(', ') + ' or ' + items[items.length - 1] : items.join(''));

  // The map's period as the server reads it (read_sheet's period_text), for the class face.
  const period = typeof data.period_text === 'string' && data.period_text ? data.period_text : null;

  /* The face's one line on Cannot tell (the last round's S1), the second
     line of the lead (r5-b J13): where the sheet takes it,
     W.cannotTellLine; a certificate plan's seen or right-or-wrong sheet,
     which takes none, keeps its own line; a two-map sheet says Cannot tell
     in its instruction. */
  // WORDING: r4h S1
  const cannotLineText = (() => {
    if (twoMap) return '';
    if (takesQ) return W.cannotTellLine;
    if (takesUnjudged) return '';
    if (seen) {
      const named = answers.filter((a) => a.value !== OE.UNJUDGED).map((a) => a.label);
      return 'A certificate needs ' + orList(named) + ' on every ' + W.one + '.';
    }
    return wrongFace ? 'A certificate needs right or wrong on every ' + W.one + '.' : '';
  })();

  /* The instruction in the Details fold (the last round's S1): a seen
     sheet's first sentence (its lead says the rest), then, where one map's
     sheet takes Cannot tell (a seen, right-or-wrong or class face), what
     Cannot tell does to the result (W.cannotTell; W.cannotTellBare, with no
     "dates shown", on a sheet that can show no pictures); a two-map sheet's
     ? is not an error rate's, so its instruction stays on the face. The
     face says Cannot tell in one line (cannotLineText). A seen sheet's
     instruction does not say again that the map's answer is hidden: the
     lead and the blind line say it (round 4 pass 2, finding 14). */
  // WORDING: r4h S1
  function instructionText() {
    const unjudged = takesQ ? ' ' + (picturesPossible ? W.cannotTell : W.cannotTellBare) : '';
    if (twoMap) return '';
    if (seen) {
      return 'For each ' + W.one + ', pick what you see there: ' + orList(answers.map((a) => a.label)) + '.'
        + unjudged;
    }
    return unjudged.trim();
  }

  /* The instruction on the face: right or wrong (the map's class is shown;
     Cannot tell or the certificate's need is in the lead), the class
     seen in the period the map covers, as the map writes classes (reader
     test 2's E/sheet-3: no "on the ground"), and a two-map sheet's, which
     names the button it has, Cannot tell, and says "there", not "on the
     ground" (r5-b J28); a seen sheet has none here (its lead says what to
     do). The class face's line is on the face only where the sheet knows
     the map's period, which it names (r5-d R08); without it the line began
     as the lead does and only asked for the period, so it is in the Details
     fold (classNoPeriod). The lead is two lines (r5-e H1; it was three), and
     the class line is a third only with the period. */
  // WORDING: r4h S1 S3; r5b J28 (the two-map sheet's)
  function faceInstructionText() {
    // r5-k K3 (r5-b J06): a sheet that gave its result takes no answer, so nothing under its notice asks for one.
    if (gaveResult) return '';
    // r5-m B2: a two-map sheet whose maps' answers were shown takes no answer (locked), so nothing asks for one.
    if (state.revealed) return '';
    const one = W.one;
    if (twoMap) return 'For each ' + one + ', pick the class you see there; press Cannot tell when you cannot judge it.';
    if (wrongFace) return 'For each ' + one + ', say whether the map is right there (the map\'s class is shown).';
    // WORDING: r5d R08 (the class face's line, on the face only with the map's period)
    if (face === 'class') return period ? W.classWhat(period) : '';
    return '';
  }

  function updateProgress() {
    const left = pending();
    counterNum.textContent = count() + '/' + n;
    counterNote.textContent = left ? left + ' change' + (left === 1 ? '' : 's') + ' not saved yet' : '';
    // r5-k K3: the answers drawn now, locked or not, in place; K2: the
    // two-map sheet's switch, locked or not. A switch locked again while the
    // maps' answers show (an answer changed after the result was sent) hides
    // them again: the editor is drawn again then, once.
    lockAnswers();
    if (syncPairLock()) { render(); return; }
    // r5-r N1, N2: "Next unlabelled" and the line of the last squares left, as the answers stand.
    updateOpen();
    if (state.stale) {
      // Nothing saves any more (r5-b J15): no Save button (before, a greyed
      // button that read "Not saved": a status dressed as a dead button).
      // Beside the counter, how many answers given here are not saved, or
      // "Not saving"; beside "Reopen the sheet", the one contained button,
      // what happened, whether an answer is lost and what to do, at body
      // size (W.changedElsewhere; W.notReadAgain where this panel's own
      // removal of a set of answers left a sheet it could not read again,
      // with everything saved before it). A Save button that had the focus as
      // it goes hands it to "Reopen the sheet", and so does the notice's
      // Remove button, which goes here too (r5-i B7: the focus fell to the
      // page after a removal whose sheet could not be read again), and so
      // does the finish button, whose row "Reopen the sheet" takes over
      // (r5-n JS-4: a press of it that found the sheet changed elsewhere left
      // the focus on the page; done holds only that button, its lines and,
      // once stale, Reopen the sheet, which focused again changes nothing).
      const hadFocus = (!saveButton.hidden && saveButton.contains(document.activeElement))
        || notice.contains(document.activeElement)
        || done.contains(document.activeElement);
      saveButton.hidden = true;
      // Hidden, it is no contained button either: the page keeps one.
      saveButton.className = '';
      savedNote.hidden = false;
      // WORDING: r5b J15
      savedNote.textContent = left ? OE.counted(left, 'answer') + ' not saved' : 'Not saving';
      counterNote.textContent = '';
      // r5-k K5: the counter's and the save's own lines go; the line beside
      // "Reopen the sheet" says what happened.
      countSaid.textContent = '';
      saveSaid.textContent = '';
      notice.replaceChildren();
      mixNote.hidden = true;
      staleLine.textContent = state.staleWhy === 'read' ? W.notReadAgain : W.changedElsewhere(left);
      if (!done.contains(reopenButton)) done.replaceChildren(staleLine, reopenButton);
      fitFoot();
      if (hadFocus && !quietSave) OE.focusIfHere(reopenButton);
      return;
    }
    // Save labels only while a change waits, a save runs or one failed; else
    // "Saved automatically" as plain text beside the counter (A/sheet-9,
    // reader test 2's B/sheet-finish-button). A Save button that had the
    // focus as it goes hands it to the finish button.
    const showSave = Boolean(left) || state.saving || state.saveFailed;
    if (!quietSave && !showSave && !saveButton.hidden && saveButton.contains(document.activeElement)) {
      OE.focusIfHere(finishControl());
    }
    saveButton.hidden = !showSave;
    savedNote.hidden = showSave;
    saveButton.className = left ? 'primary' : '';
    saveButton.textContent = state.saving ? 'Saving...' : 'Save labels';
    // Busy while saving: aria-disabled (save() ignores a press then), so the button keeps the focus.
    saveButton.setAttribute('aria-disabled', String(state.saving));
    // r5-k K5: a change that waits or a save under way empties the save's own line unless it says a failure (Save
    // labels stays shown with it), so no earlier "Every change is saved to ..." stays under Save labels.
    if ((left || state.saving) && !saveSaid.classList.contains('error')) saveSaid.textContent = '';
    // Two sets of answers on one sheet (state.mix: the server's word that
    // the estimate would refuse it). The notice says so plainly, with a
    // button that removes the set in the way, each set named by what it
    // holds (answersOf; r5-b J21: "Clear the wrong column" read as "the
    // incorrect one", and did not say which answers go). The server's own
    // sentence is in the Details fold (mixNote). A sheet whose answers are
    // locked (r5-k K3: one that gave its result, the only one locked here)
    // keeps the notice and its fold line but offers no Remove button, whose
    // press would clear a whole set of its answers. r5-m B6: on a sheet that
    // gave its result the notice only says what the sheet now holds ("so no
    // result can come from it" read, under the graded notice, as if the
    // result given did not exist); the graded notice and the footer say that
    // a new result needs a fresh sheet.
    notice.replaceChildren();
    mixNote.hidden = !state.mix;
    mixNote.textContent = state.mix ? 'The estimate would refuse this sheet: ' + state.mix.problem + '.' : '';
    if (state.mix) {
      const mix = state.mix;
      const kept = mix.column === 'wrong' ? 'reference_class' : 'wrong';
      // WORDING: r5m B6
      // WORDING: r5b J21
      notice.append(OE.el('div', {}, OE.el('span', { class: 'error' }, gaveResult
        ? 'This sheet now holds two sets of answers. '
        : 'This sheet holds two sets of answers, so no result can come from it. '),
      mix.column && !locked() ? OE.el('button', { class: 'small', onclick: () => clearAndOpen(mix.column, face) },
        'Remove the ' + answersOf(mix.column) + ', keep the ' + answersOf(kept)) : null));
    }
    updateFinish();
    fitFoot();
  }

  /* The classes a window's list offers, in order: the face's classes; on a
     two-map sheet unblinded, the window's pair first; then ? where the face
     takes it (a class map's class face too, the last round's S4). */
  function classOptions(h) {
    if (!twoMap) return takesQ ? [...listed, OE.UNJUDGED] : listed;
    const extra = state.extra.get(h) || {};
    const pair = !state.blind && state.revealed && Array.isArray(extra.pair) ? extra.pair.filter(Boolean) : [];
    const options = pair.length ? [...new Set([...pair, ...listed])] : listed;
    return takesQ ? [...options, OE.UNJUDGED] : options;
  }

  /* r5-k K3 (the owner's O19): a sheet that already gave its result
     (data.graded) takes no answer, and neither does one that says it is not
     saving (state.stale: a save was refused because the sheet changed under
     the page; its footer says why and what to do, and an answer given then
     would only be lost). Locked, every answer control of the face and of the
     list is aria-disabled with the greyed busy look and keeps the focus
     (lockAnswers), no press, key or menu sets or clears an answer (blocked,
     and each control's own guard), and Previous, Next, Pictures, Street map,
     the list and the finish button work as before. A class being typed when
     the lock comes keeps its text, read only. It unlocks once neither holds
     (an editor drawn again from a fresh read starts unlocked). r5-m B2: nor
     does a two-map sheet once the reviewer has seen what the two maps say
     (state.revealed, set by the switch's first untick after the finish
     press), so no answer changes after the maps' answers were shown; it
     stays locked when the switch is ticked again (the answers were seen).
     A lock in the view only: an editor drawn again from a fresh read starts
     unlocked, as above. */
  const gaveResult = Boolean(data.graded && typeof data.graded === 'object');
  const locked = () => gaveResult || state.stale || state.revealed;
  /* While a column is cleared, the editor is about to be drawn again from
     the file: no answer is taken (the area is inert and aria-busy
     meanwhile, busyArea), so none is dropped unsaved; nor while the answers
     are locked (r5-k K3). */
  const blocked = () => state.clearing || locked();
  /* The answer controls as drawn now (the face's answers, its list of more
     classes, Other class and a typed class with its OK; each list row's
     answer cell), marked locked or not in place, so nothing is drawn again
     and a typed class keeps its text: aria-disabled (the busy look of
     common.js), described by the line that says why (the graded notice; on
     a two-map sheet whose maps' answers were shown, r5-m B2, the lead's
     first line, which then always says the answers are locked, the switch
     unticked (pairShown) or ticked again (pairSeen), and on a sheet with no
     pictures and no places too (render); or the footer's line beside
     "Reopen the sheet"), a field read only. Run by
     updateProgress, so after every render and when the sheet stops saving. */
  function lockAnswers() {
    const lock = locked();
    const why = gaveResult && gradedNote ? gradedNote : state.revealed && pairWhat ? pairWhat : staleLine;
    const controls = [...faceAnswerArea.querySelectorAll('button, select, input'),
      ...body.querySelectorAll('td:last-child button, td:last-child select, td:last-child input')];
    for (const c of controls) {
      if (lock) {
        c.setAttribute('aria-disabled', 'true');
        c.setAttribute('aria-describedby', why.id);
        c.setAttribute('data-oe-locked', '');
        if (c.tagName === 'INPUT') c.readOnly = true;
      } else if (c.hasAttribute('data-oe-locked')) {
        c.removeAttribute('aria-disabled');
        c.removeAttribute('aria-describedby');
        c.removeAttribute('data-oe-locked');
        if (c.tagName === 'INPUT') c.readOnly = false;
      }
    }
  }
  function busyArea(on) {
    area.inert = on;
    if (on) area.setAttribute('aria-busy', 'true'); else area.removeAttribute('aria-busy');
  }

  function setLabel(h, value) {
    if (blocked()) return;
    if (state.labels.get(h) === value) return;
    // r5-k K2: a real change undoes the finish press: a two-map sheet's switch stays locked until a new press
    // Claude takes (pairLocked), and its line asks for that press again (a message shown to copy is out of date).
    resultSent = false;
    resultShown = false;
    state.labels.set(h, value);
    state.dirty.add(h);
    updateProgress();
    schedule();
  }

  // The data-focus key of an answer button.
  const answerFocus = (value, h) => (value === OE.UNJUDGED ? 'au' : 'a' + value) + ':' + h;

  function labelCell(w) {
    const h = w.handle;
    const k = place(h);
    const current = state.labels.get(h) || '';
    if (closed) {
      // One joined toggle of the face's answers; a second press clears the
      // answer. A press that sets an answer moves on to the next window not
      // yet answered. On a seen sheet the server works out wrong.
      const pick = (value) => {
        if (blocked()) return;  // the editor is about to be drawn again: no answer now
        const next = (state.labels.get(h) || '') === value ? '' : value;
        setLabel(h, next);
        if (next) advance(h, answerFocus(value, h)); else render(answerFocus(value, h));
      };
      return OE.el('div', {
        class: 'toggle small', role: 'group', tabindex: '-1', 'data-focus': 'g:' + h,
        'aria-label': seen ? W.seenIn(k) : W.rightIn(k),
      }, answers.map((a) => OE.el('button', {
        type: 'button', 'data-focus': answerFocus(a.value, h), 'aria-pressed': String(current === a.value),
        'aria-keyshortcuts': keysOf(answers, a.value), onclick: () => pick(a.value),
      }, a.label)));
    }
    if (state.other === h) {
      const input = OE.el('input', {
        type: 'text', 'aria-label': W.otherIn(k), maxlength: '200',
        placeholder: 'class seen', value: current, 'data-focus': 'o:' + h,
      });
      const accept = () => {
        if (locked()) return;  // r5-k K3: locked, the field and its text stay as they are
        state.other = null;
        const typed = input.value.replace(/[\r\n]+/g, ' ').trim().slice(0, 200);
        if (typed && typed !== OE.UNJUDGED) {
          setLabel(h, typed);
          advance(h, 's:' + h);  // a typed class accepted (Enter or OK) moves on
        } else render('s:' + h);
      };
      input.addEventListener('keydown', (e) => {
        if (e.isComposing || e.keyCode === 229) return;  // an IME's Enter commits its text, nothing more
        // r5-k K3: locked, Enter and Escape do nothing, so what was typed is not lost.
        if (locked()) { if (e.key === 'Enter' || e.key === 'Escape') e.preventDefault(); return; }
        if (e.key === 'Enter') accept();
        if (e.key === 'Escape') { state.other = null; render('s:' + h); }
      });
      return OE.el('div', { class: 'row' }, input,
        OE.el('button', { class: 'small', 'data-focus': 'ok:' + h, 'aria-label': W.okIn(k), onclick: accept }, 'OK'));
    }
    const options = classOptions(h);
    const select = OE.el('select', {
      'aria-label': W.seenIn(k), 'data-focus': 's:' + h,
      'aria-keyshortcuts': answerKeys(w).map((a) => a.keys[0]).join(' ') || null,
      // r5-k K3: locked, a press does not open the menu (onKey keeps its keys
      // from changing it), and a change that still gets through goes back.
      onmousedown: (e) => { if (locked()) e.preventDefault(); },
      onchange: () => {
        if (locked()) { select.value = state.labels.get(h) || ''; return; }
        if (select.value === OE.OTHER) { state.other = h; render('o:' + h); return; }
        // Browsing the list with the keyboard changes it too: only a pick moves on.
        const browsing = Boolean(lastSelectKey && lastSelectKey.focus === 's:' + h && lastSelectKey.browse);
        lastSelectKey = null;
        setLabel(h, select.value);
        if (select.value && !browsing) advance(h, 's:' + h);
      },
    },
    // A row with no answer says so, in the word of the Unlabelled filter and
    // the counter (r5-j B7: "(class seen)" read like an answer given).
    // WORDING: r5j B7
    OE.el('option', { value: '' }, '(not labelled yet)'),
    // Cannot tell in the words of the face's button (r5-d R13: it read
    // "? (cannot tell)"); its value stays OE.UNJUDGED.
    // WORDING: r5d R13
    options.map((c) => OE.el('option', { value: c }, c === OE.UNJUDGED ? 'Cannot tell (?)' : c)));
    if (current && !options.includes(current)) {
      select.append(OE.el('option', { value: current }, current));
    }
    select.append(OE.el('option', { value: OE.OTHER }, 'Other class...'));
    select.value = current;
    return select;
  }

  /* The focused row control, before render() rebuilds the rows: its key and
     its row's place in the table. */
  function focusedControl() {
    const active = document.activeElement;
    if (!active || !body.contains(active) || !active.dataset.focus) return null;
    const tr = active.closest('tr');
    return { key: active.dataset.focus, place: tr ? [...body.children].indexOf(tr) : -1 };
  }

  function restoreFocus(was) {
    if (!was) return;
    let target = body.querySelector('[data-focus="' + CSS.escape(was.key) + '"]');
    if (!target && was.place >= 0) {
      // The row left the filtered list: the same kind of control in the row now at its place.
      const rowsNow = [...body.children];
      const tr = rowsNow[Math.min(was.place, rowsNow.length - 1)];
      const kind = was.key.split(':')[0];
      target = tr ? tr.querySelector('[data-focus^="' + kind + ':"]') || tr.querySelector('[data-focus]') : null;
    }
    // Only while the panel has the focus: never on load, never out of the chat.
    OE.focusIfHere(target);
  }

  /* Whether a window holds an answer on the face. */
  const isLabelledRow = (w) => answered(state.labels.get(w.handle) || '');
  /* The windows that show: the order of their handles, under the filter. */
  const visibleRows = () => windows.filter((w) => state.filter === 'any'
    || (state.filter === 'labelled') === isLabelledRow(w));

  /* The first window after h, in the order shown and wrapping round, that
     holds no answer; the filter is ignored. null when every one has one. */
  function nextUnanswered(h) {
    const at = windows.findIndex((w) => w.handle === h);
    for (let step = 1; step < windows.length; step += 1) {
      const w = windows[(at + step) % windows.length];
      if (!isLabelledRow(w)) return w;
    }
    return null;
  }

  /* r5-r: the windows with no answer, in the order of the handles. */
  const openRows = () => windows.filter((w) => !isLabelledRow(w));

  /* r5-r: a move asked for from outside the face (the footer's jump, a
     window named in the counter's row) gives the face the focus, so the
     answers' keys work at once; a press inside the face ("Next unlabelled",
     U) keeps the focus where it is. Where a click gives a button no focus
     (the focus is then on no control), the face takes it too. The window
     shows as Next shows one (pageBy, by as many places as lie between; none
     when it is the one shown, which is drawn again and its heading brought
     into view). */
  function showFromAnywhere(h) {
    pageBy(place(h) - place(state.at));
    if (!faceEl.contains(document.activeElement)) { OE.focusIfHere(faceEl); headInView(); }
  }

  /* r5-r N1: "Next unlabelled", its key U (inFace true), and the jump at the
     end of the grey finish button's line (N2). On the face: the next window
     with no answer after the one shown, wrapping round to the first from the
     top (nextUnanswered), or the one shown when it alone has none: the
     button (grey then, updateOpen) and U then do nothing, and the footer's
     jump brings its heading back into view. It shows as Next shows a window
     (showFromAnywhere: the focus stays on the button pressed inside the
     face, which stays while a window has no answer; from the footer the
     face takes the focus). Showing is not answering: it
     works on a locked sheet, as Previous and Next do. In the list (only the
     footer's jump is there): the first row with no answer takes the focus,
     at its answers, as after an answer (advance); where the filter hides it
     (Labelled), that window shows on the face. */
  function jumpOpen(inFace) {
    const left = openRows();
    if (!left.length) return;
    if (state.view === 'face') {
      const next = nextUnanswered(state.at) || left[0];
      if (inFace && next.handle === state.at) return;
      showFromAnywhere(next.handle);
      return;
    }
    const first = left[0];
    if (!visibleRows().includes(first)) { setView('face', first.handle); return; }
    render(firstAnswerFocusKey(first));
    followPictures(first.handle);
  }

  /* r5-r N2: a named window opens: on the face it shows, as Next shows one,
     and the face takes the focus (showFromAnywhere: the press came from the
     footer); from the list it shows on the face, as a press of its row's
     number does (setView, which gives the face the focus). */
  function openRow(h) {
    if (state.view === 'face') { showFromAnywhere(h); return; }
    setView('face', h);
  }

  /* r5-r: "Next unlabelled" and the line of the last windows left, as the
     answers stand (run by updateProgress, so after every answer, every
     render and every take-in of a save made elsewhere).
     N1: the button shows only while a window has no answer, and the keys
     line offers its key only then (faceHintText, written here too, as a
     save taken in may not draw the sheet again). When
     it goes while it holds the focus (the last answer taken in from
     elsewhere), the face takes the focus, where it waits after an answer.
     It is grey (aria-disabled, which keeps the focus) while the window shown
     is the only one with no answer: there is no other to show.
     N2: while 1 to 3 windows are left and at least one is answered (the
     last few of a sheet, not the whole of a fresh sheet of three), the
     counter's row ends with "Unlabelled: Square 29" or "Unlabelled: Squares
     7, 29 and 80", each window a text button named "Go to square 29" that
     opens it (openRow). The line is drawn again only when its windows
     change, so a button of it keeps the focus across an answer elsewhere;
     when it goes while it holds the focus, the finish button takes it. A
     window's place only: nothing of what the map says there. */
  let leftShown = '';
  function updateOpen() {
    const left = openRows();
    const show = left.length > 0;
    if (nextOpenButton.hidden === show) {
      const had = !show && nextOpenButton.contains(document.activeElement);
      nextOpenButton.hidden = !show;
      if (state.view === 'face') { hint.textContent = faceHintText(); hint.hidden = !hint.textContent; }
      if (had && !quietSave) OE.focusIfHere(faceEl);
      // The jump of the footer's line goes with that line (updateFinish): the finish button takes its focus.
      if (!show && goOpenLink.contains(document.activeElement) && !quietSave) OE.focusIfHere(finishControl());
    }
    const only = state.view === 'face' && left.length === 1 && left[0].handle === state.at;
    nextOpenButton.setAttribute('aria-disabled', String(only));
    const few = left.length >= 1 && left.length <= 3 && left.length < n;
    const key = few ? left.map((w) => w.handle).join(' ') : '';
    if (key === leftShown) return;
    const had = leftLine.contains(document.activeElement);
    leftShown = key;
    leftLine.hidden = !few;
    if (!few) {
      leftLine.replaceChildren();
      if (had && !quietSave) OE.focusIfHere(finishControl());
      return;
    }
    const link = (w, words) => OE.el('button', {
      class: 'link', style: LINE_LINK, 'data-focus': 'left:' + w.handle, 'aria-label': W.goTo(place(w.handle)),
      onclick: () => openRow(w.handle),
    }, words);
    const parts = [W.leftLead];
    if (left.length === 1) {
      parts.push(link(left[0], W.leftOne(place(left[0].handle))));
    } else {
      parts.push(W.leftMany);
      left.forEach((w, i) => {
        if (i) parts.push(i === left.length - 1 ? ' and ' : ', ');
        parts.push(link(w, String(place(w.handle))));
      });
    }
    leftLine.replaceChildren(...parts);
    if (had && !quietSave) OE.focusIfHere(leftLine.querySelector('button'));
  }

  /* The data-focus key of the control a window's answer is given at, once
     the focus moves on to it: its group of answers (tabindex -1, so Tab
     still reaches each answer), never an answer button, so Enter or Space
     alone sets nothing and a held Enter cannot answer row after row; the
     row's keys work there. The class seen: the row's list. */
  const firstAnswerFocusKey = (w) => (closed ? 'g:' : 's:') + w.handle;

  /* After an answer on window h in the list: the focus goes to the next
     window not yet answered (its group of answers, or its list), and the
     pictures follow it. When the filter hides that window, the focus stays
     on the control the answer came from (stay); with every window answered
     it stays, and goes to the finish button once that is ready. */
  function advance(h, stay) {
    state.finishFrom = null;
    const next = nextUnanswered(h);
    if (!next) {
      render(stay);
      state.finishFrom = stay;
      updateFinish();
      return;
    }
    if (!visibleRows().includes(next)) { render(stay); return; }
    render(firstAnswerFocusKey(next));
    followPictures(next.handle);
  }

  /* The pictures follow the focus in the list to window h 300 ms after the
     answer (a quick run of answers asks once), when the pane is open or
     pictures are on, and not over the public test scene. The server answers
     with the question, not pictures, until the user has switched them on. */
  function followPictures(h) {
    clearTimeout(picturesTimer);
    const follow = () => wrap.isConnected && state.view === 'list' && !state.test
      && (state.shown !== null || state.picturesOn);
    if (!follow()) return;
    picturesTimer = setTimeout(() => { if (follow()) loadPictures(h); }, 300);
  }

  /* The answers a key sets on a row of the list, each with its keys (upper case). */
  function answerKeys(w) {
    if (closed) return answers;
    // The class seen: 1 to 9 for the n-th class of the row's list; ? where the
    // face takes Cannot tell (a two-map sheet, and a class map's: S4).
    const classes = classOptions(w.handle).filter((c) => c !== OE.UNJUDGED).slice(0, 9)
      .map((c, i) => ({ value: c, label: c, keys: [String(i + 1)] }));
    return takesQ ? [...classes, { value: OE.UNJUDGED, label: 'Cannot tell', keys: [OE.UNJUDGED] }] : classes;
  }

  /* The line under the instruction in the list: the answers' keys, as the face has them. */
  function listHintText() {
    const after = W.listAfter;
    if (closed) {
      const list = answers.filter((a) => a.keys.length).map((a) => a.keys[0] + ' ' + a.label.toLowerCase());
      return list.length ? 'Keys: ' + list.join(', ') + '.' + after : '';
    }
    const most = windows.reduce((m, w) => Math.max(m,
      classOptions(w.handle).filter((c) => c !== OE.UNJUDGED).length), 0);
    const parts = [];
    if (most === 1) parts.push('1 picks the first class');
    else if (most > 9) parts.push('1 to 9 pick one of the first nine classes');
    else if (most > 1) parts.push('1 to ' + most + ' pick a class in the list\'s order');
    if (takesQ) parts.push('? cannot tell');
    return parts.length ? 'Keys: ' + parts.join(', ') + '.' + after : '';
  }

  /* The line under the instruction on the face: each answer's key, in the
     server's order, and the arrow keys. */
  function faceHintText() {
    const named = answers.filter((a) => a.keys.length)
      .map((a) => a.keys[0] + ' ' + (closed || a.value === OE.UNJUDGED ? a.label.toLowerCase() : a.label));
    // r5-r N1: the key of "Next unlabelled", where no answer has it, and
    // only while a window has no answer (the button is gone after, and the
    // key does nothing; updateOpen writes the line again when that changes).
    return 'Keys: ' + (named.length ? named.join(', ') + '; the' : 'the') + W.faceKeys
      + (openKey && openRows().length ? W.nextOpenKey : '');
  }

  const BROWSE = new Set(['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Home', 'End', 'PageUp', 'PageDown']);
  /* A key on a row (the rows' one listener): sets the row's answer, never
     clears it, and moves on as a press does. Modified, held and composing
     keys and typing in a field are left alone. A held key answers nothing:
     a repeated keydown sets no answer, and a repeated Enter or Space is kept
     from pressing the focused button again (the browser presses a button
     on each Enter, a repeated one too). Enter or Space on a row's group of
     answers does nothing (no scroll either). A key on a list is noted, so
     its change event knows whether the user was browsing it. */
  function onKey(e) {
    const t = e.target;
    const k = typeof e.key === 'string' ? e.key : '';  // an autofill keydown has no key
    if (t && t.tagName === 'SELECT') {
      lastSelectKey = { focus: t.dataset.focus, browse: BROWSE.has(k) || (k.length === 1 && k !== ' ') };
    }
    // r5-k K3: locked, no key changes or opens a row's menu (Tab still moves on).
    if (locked() && t && t.tagName === 'SELECT' && !e.ctrlKey && !e.metaKey
        && (BROWSE.has(k) || k.length === 1 || k === 'Enter' || k === 'F4')) {
      e.preventDefault();
      return;
    }
    const typing = !t || t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable;
    const onGroup = Boolean(t && t.dataset && String(t.dataset.focus || '').startsWith('g:'));
    if (!typing && (k === 'Enter' || k === ' ') && (e.repeat || onGroup)) {
      e.preventDefault();
      return;
    }
    if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey || e.repeat || e.isComposing) return;
    if (!t || !t.closest || t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable) return;
    const tr = t.closest('tr[data-handle]');
    const w = tr ? byHandle.get(tr.dataset.handle) : null;
    if (!w) return;
    const pressed = k.length === 1 ? k.toUpperCase() : '';
    const hit = pressed ? answerKeys(w).find((a) => a.keys.includes(pressed)) : null;
    if (!hit) return;
    e.preventDefault();
    if (blocked()) return;  // the editor is about to be drawn again: no answer now
    if (t.tagName === 'SELECT') lastSelectKey = null;  // a digit is an answer, not browsing
    if (state.other === w.handle) state.other = null;  // the key's class replaces one being typed
    setLabel(w.handle, hit.value);
    advance(w.handle, t.dataset.focus || firstAnswerFocusKey(w));
  }

  /* The face's one key listener. An answer's key (the server's: P, A, ? and
     1 to 9; R, W and ?; 1 to 9 for a class) sets the window's answer, never
     clears it, and the next window not yet labelled shows; the left and
     right arrow keys move to the window before or after. Modified, held and
     composing keys, and keys typed into a field or a list, are left alone;
     a held Enter or Space presses nothing again, and Enter or Space on the
     face itself does nothing (no scroll either). */
  function onFaceKey(e) {
    const t = e.target;
    const k = typeof e.key === 'string' ? e.key : '';
    const typing = !t || t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable;
    if (typing) return;
    if ((k === 'Enter' || k === ' ') && (e.repeat || t === faceEl)) {
      e.preventDefault();
      return;
    }
    if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey || e.repeat || e.isComposing) return;
    if (k === 'ArrowLeft' || k === 'ArrowRight') {
      // Inside a control that scrolls sideways, the arrow keys scroll it;
      // elsewhere on the face they page (r5-k K4: the dates wrap now, so they
      // no longer scroll sideways and their arrow keys page too). Paging is
      // not answering: it works on a locked sheet (K3).
      if (scrollsSideways(t)) return;
      e.preventDefault();
      pageBy(k === 'ArrowLeft' ? -1 : 1);
      return;
    }
    const pressed = k.length === 1 ? k.toUpperCase() : '';
    const hit = pressed && state.at ? answers.find((a) => a.keys.includes(pressed)) : null;
    // r5-r N1: U, where no answer has that key (openKey), shows the next
    // window with no answer, as a press of "Next unlabelled" does; with every
    // window answered it does nothing. Showing is not answering: it works on
    // a locked sheet, as the arrow keys do.
    if (!hit && openKey && pressed === openKey && state.at) {
      e.preventDefault();
      jumpOpen(true);
      return;
    }
    if (!hit) return;
    e.preventDefault();
    if (blocked()) return;  // the editor is about to be drawn again: no answer now
    state.other = null;
    setLabel(state.at, hit.value);
    faceAdvance();
  }

  /* Whether a key's target is in a control inside the face that scrolls
     sideways now (the dates strip inline, wider than the panel): its arrow
     keys are its own. */
  function scrollsSideways(t) {
    for (let el = t; el && el !== faceEl && el.nodeType === 1; el = el.parentElement) {
      const x = getComputedStyle(el).overflowX;
      if ((x === 'auto' || x === 'scroll') && el.scrollWidth > el.clientWidth) return true;
    }
    return false;
  }

  /* The instructions (on the face, and in the fold) and the keys, then the
     view shown: the face (renderFace) or the list's rows (renderList). A
     two-map sheet's first line says whether the maps' answers are hidden or
     shown, by extraLines' own test (r5-b J28); once they were shown it
     says the answers are locked, the switch ticked again or not (r5-m B2:
     pairSeen when ticked), and no line on the face asks for an answer. A
     two-map sheet's switch locked again first hides what it showed (r5-k
     K2, syncPairLock). */
  function render(focusKey) {
    syncPairLock();
    if (pairWhat) pairWhat.textContent = !state.revealed ? pairHidden : state.blind ? pairSeen : pairShown;
    faceInstruction.textContent = faceInstructionText();
    faceInstruction.hidden = !faceInstruction.textContent;
    // r5-m B2 (r5-b J06): the line that asks for a rule before the first
    // answer goes once the maps' answers were shown (the answers are locked
    // then, for good in this panel), as on a graded sheet.
    if (noRuleLine && state.revealed) noRuleLine.hidden = true;
    instruction.textContent = instructionText();
    instruction.hidden = !instruction.textContent;
    hint.textContent = state.view === 'face' ? faceHintText() : listHintText();
    hint.hidden = !hint.textContent;
    if (state.view === 'face') renderFace(focusKey);
    else renderList(focusKey);
  }

  /* What the face shows beside a window's answer: the map's class on the
     right-or-wrong face only; on a two-map sheet, once the reviewer unticked
     blind, the two maps' classes. A face that labels the class seen never
     shows the map's answer (the owner's O1), and no face a stratum (F11). */
  function extraLines(h) {
    const extra = state.extra.get(h) || {};
    const unblinded = !state.blind && state.revealed;
    const lines = [];
    if (wrongFace && extra.mapClass !== null) {
      lines.push(OE.el('div', {}, 'Map class: ', OE.el('strong', {}, extra.mapClass || '(none)')));
    }
    if (twoMap && unblinded && Array.isArray(extra.pair)) {
      lines.push(OE.el('div', {}, 'Map A / Map B: ' + (extra.pair[0] || '') + ' / ' + (extra.pair[1] || '')));
    }
    return lines;
  }

  /* The face's answers for window w: a button for each of the first nine
     answers, with its key ("Present (P)"), then Cannot tell where the face
     takes it, always a button, never only in the list (the last round's S4,
     for a class map of many classes), the rest in a list; the window's own
     answer when it is none of them (a class typed in); and on a face that
     takes any class, "Other class..." (the group's last cell, r5-e H3) or
     the field to type it. A name may break after each underscore
     (OE.softBreaks, r5-f F14: "evergreen_needleleaf_woodland" was cut in
     mid-word by the cell's overflow-wrap, which stays for a word with none);
     its text is as the server wrote it. */
  function faceAnswers(w) {
    const h = w.handle;
    const k = place(h);
    const current = state.labels.get(h) || '';
    const own = current && !answers.some((a) => a.value === current);
    const named = answers.filter((a) => a.value !== OE.UNJUDGED);
    const cannotTell = answers.filter((a) => a.value === OE.UNJUDGED);
    const group = OE.el('div', {
      class: 'answers row', role: 'group',
      'aria-label': wrongFace ? W.rightIn(k) : W.seenIn(k),
    }, [...named.slice(0, 9), ...cannotTell].map((a) => OE.el('button', {
      type: 'button', 'data-focus': answerFocus(a.value, h), 'aria-pressed': String(current === a.value),
      'aria-keyshortcuts': a.keys.join(' ') || null, onclick: () => facePick(a.value),
    }, OE.softBreaks(a.label + (a.keys.length ? ' (' + a.keys[0] + ')' : '')))),
    own ? OE.el('button', { type: 'button', 'data-focus': 'x:' + h, 'aria-pressed': 'true', onclick: () => facePick(current) }, OE.softBreaks(current)) : null);
    const rest = named.slice(9);
    if (rest.length) {
      const list = OE.el('select', {
        'aria-label': W.moreIn(k), 'data-focus': 'r:' + h,
        onchange: () => {
          // r5-k K3: locked, a change that gets through goes back to the answer given.
          if (locked()) { list.value = rest.some((a) => a.value === current) ? current : ''; return; }
          const browsing = faceListKey;
          faceListKey = false;
          if (!list.value) { render('r:' + h); return; }
          if (!browsing) { facePick(list.value); return; }
          // Browsing the list with the keyboard sets the answer and stays.
          setLabel(h, list.value);
          for (const b of group.querySelectorAll('button[aria-pressed]')) b.setAttribute('aria-pressed', 'false');
        },
      }, OE.el('option', { value: '' }, '(more classes)'), rest.map((a) => OE.el('option', { value: a.value }, a.label)));
      list.addEventListener('keydown', (e) => {
        const key = typeof e.key === 'string' ? e.key : '';
        // r5-k K3: locked, no key changes or opens the list (Tab still moves on).
        if (locked() && !e.ctrlKey && !e.metaKey
            && (BROWSE.has(key) || key.length === 1 || key === 'Enter' || key === 'F4')) {
          e.preventDefault();
          return;
        }
        faceListKey = BROWSE.has(key) || (key.length === 1 && key !== ' ');
      });
      list.addEventListener('pointerdown', () => { faceListKey = false; });
      // r5-k K3: locked, a press does not open the list.
      list.addEventListener('mousedown', (e) => { if (locked()) e.preventDefault(); });
      list.value = rest.some((a) => a.value === current) ? current : '';
      group.append(list);
    }
    const parts = [group];
    if (!closed && state.other === h) {
      const input = OE.el('input', {
        type: 'text', 'aria-label': W.otherIn(k), maxlength: '200',
        placeholder: 'class seen', value: own ? current : '', 'data-focus': 'o:' + h,
      });
      const accept = () => {
        if (locked()) return;  // r5-k K3: locked, the field and its text stay as they are
        state.other = null;
        const typed = input.value.replace(/[\r\n]+/g, ' ').trim().slice(0, 200);
        if (typed && typed !== OE.UNJUDGED) {
          setLabel(h, typed);
          faceAdvance();  // a typed class accepted (Enter or OK) moves on
        } else render('other:' + h);
      };
      input.addEventListener('keydown', (e) => {
        if (e.isComposing || e.keyCode === 229) return;  // an IME's Enter commits its text, nothing more
        // r5-k K3: locked, Enter and Escape do nothing, so what was typed is not lost.
        if (locked()) { if (e.key === 'Enter' || e.key === 'Escape') e.preventDefault(); return; }
        if (e.key === 'Enter') { e.preventDefault(); accept(); }
        if (e.key === 'Escape') { e.preventDefault(); state.other = null; render('other:' + h); }
      });
      parts.push(OE.el('div', { class: 'row' }, input,
        OE.el('button', { class: 'small', 'data-focus': 'ok:' + h, 'aria-label': W.okIn(k), onclick: accept }, 'OK')));
    } else if (!closed) {
      // The last cell of the answers' group, as large as every answer, with
      // a dashed border (common.js .oe-other; r5-e H3: before, a small text
      // link on a line of its own under the answers).
      // r5-k K3: locked, its press opens no field.
      group.append(OE.el('button', {
        type: 'button', class: 'oe-other', 'data-focus': 'other:' + h, onclick: () => { if (!locked()) { state.other = h; render('o:' + h); } },
      }, 'Other class...'));
    }
    return parts;
  }

  /* The face for the window it shows (state.at): its heading and the pager,
     its pictures, then what it shows beside its answer and its answers
     (r5-k K4). focusKey,
     after the user's action: the control (its data-focus key), else the face
     itself ('face'). Without one, a focus lost with a rebuilt control goes to
     the face. Never on load, and only while the panel has the focus
     (OE.focusIfHere). */
  function renderFace(focusKey) {
    const hadFocus = faceEl.contains(document.activeElement);
    const h = state.at;
    const w = h ? byHandle.get(h) : null;
    if (!w) { updateProgress(); return; }
    const k = place(h);
    faceTitle.textContent = W.title(k, n);
    prevButton.setAttribute('aria-disabled', String(k <= 1));
    nextButton.setAttribute('aria-disabled', String(k >= n));
    const extras = extraLines(h);
    faceExtra.replaceChildren(...extras);
    faceExtra.hidden = !extras.length;
    faceAnswerArea.replaceChildren(...faceAnswers(w));
    if (facePicturesButton) {
      facePicturesButton.setAttribute('data-focus', 'i:' + h);
      facePicturesButton.setAttribute('aria-label', W.pictures(k));
      facePicturesButton.setAttribute('aria-expanded', String(state.shown === h));
    }
    if (faceMapButton) {
      faceMapButton.setAttribute('data-focus', 'm:' + h);
      faceMapButton.setAttribute('aria-label', W.openMapOf(k));
      faceMapButton.setAttribute('aria-disabled', String(linking.has(h)));
    }
    if (preview.parentNode !== facePictures) facePictures.append(preview);
    renderPreview();
    updateProgress();
    let target = null;
    if (focusKey === 'face') target = faceEl;
    else if (focusKey) target = faceEl.querySelector('[data-focus="' + CSS.escape(focusKey) + '"]') || faceEl;
    else if (hadFocus && !faceEl.contains(document.activeElement)) target = faceEl;
    OE.focusIfHere(target);
  }

  /* Show window h on the face. While the pane is open or pictures are on,
     its pictures show at once: what the server gave for it, else
     "Loading", asked for 300 ms later (a quick run of moves asks once), so
     the face never shows one window's pictures under another's heading. */
  function goTo(h, focusKey) {
    if (!h || !byHandle.has(h)) return;
    state.at = h;
    if (state.other !== h) state.other = null;
    state.test = null;
    clearTimeout(picturesTimer);
    if (state.shown !== null || state.picturesOn) {
      state.shown = h;
      if (!state.images.has(h)) {
        picturesTimer = setTimeout(() => {
          if (wrap.isConnected && state.view === 'face' && state.at === h && state.shown === h
              && !state.images.has(h)) loadPictures(h);
        }, 300);
      }
    }
    render(focusKey);
  }

  /* Previous or Next, and the left and right arrow keys: the window before
     or after the one shown, in the order of the handles; nothing past
     either end. The new square's heading is then brought into view
     (headInView: an arrow key pressed down at its answers); a press of
     Previous or Next, which sit in that heading's row, moves nothing. */
  function pageBy(step) {
    const at = windows.findIndex((w) => w.handle === state.at);
    const to = at + step;
    if (at < 0 || to < 0 || to >= n) return;
    goTo(windows[to].handle);
    headInView();
  }

  /* r5-k K4 (the owner's O8): the pictures stand above the answers and the
     dates wrap, so the face is taller than the view; after an answer that
     moves on, or a move by the arrow keys, the new square's heading (and the
     pictures under it) is brought into view, the least scroll that shows it
     ('nearest': none while it is in view), so its answers are not reached
     with its pictures out of view above. Inline the host grows the frame to
     its content, so the scroll may be the host's page's. */
  function headInView() {
    const top = faceEl.querySelector('.face-top');
    if (top && typeof top.scrollIntoView === 'function') top.scrollIntoView({ block: 'nearest' });
  }

  /* An answer pressed on the face: a press on the answer given clears it and
     stays; any other sets it, and the next window not yet labelled shows. */
  function facePick(value) {
    const h = state.at;
    if (!h || blocked()) return;  // blocked: the editor is about to be drawn again
    const next = (state.labels.get(h) || '') === value ? '' : value;
    state.other = null;
    setLabel(h, next);
    if (next) faceAdvance(); else render(answerFocus(value, h));
  }

  /* After an answer on the face: the next window not yet labelled, in the
     order of the handles and wrapping round, shows, and the focus stays on
     the face, its heading brought into view (headInView, r5-k K4); with
     every window labelled the face stays where it is, and the finish button
     takes the focus once it is ready. */
  function faceAdvance() {
    state.finishFrom = null;
    const next = nextUnanswered(state.at);
    if (!next) {
      render('face');
      state.finishFrom = 'face';
      updateFinish();
      return;
    }
    goTo(next.handle, 'face');
    headInView();
  }

  /* One window or List, at the user's press, or a list row's number (at),
     which shows that window on the face; kept for the editors drawn after
     it on this page (OE.sheetView). On the face an open pane goes to its
     foot: a question kept for the window shown is brought into view (r5-f
     F01); the switch to List leaves the scroll as it is, by the table. */
  function setView(v, at) {
    if (v !== 'face' && v !== 'list') return;
    state.view = v;
    OE.sheetView = v;
    [...viewToggle.querySelectorAll('button')].forEach((b, i) => b.setAttribute('aria-pressed', String(['face', 'list'][i] === v)));
    filterBox.hidden = v !== 'list';
    if (v === 'face') {
      area.replaceChildren(faceEl);
      goTo(at || state.at, at ? 'face' : null);
      if (state.shown !== null && state.shown === state.at) questionInView(state.images.get(state.at));
      return;
    }
    clearTimeout(picturesTimer);
    state.test = null;
    area.replaceChildren(work);
    if (preview.parentNode !== work) work.append(preview);
    render();
    // The pane, headed by its window now; pictures the face had not asked for yet are asked for.
    if (state.shown !== null && !state.images.has(state.shown)) loadPictures(state.shown);
    else renderPreview();
  }

  /* The list's rows: "#" (the window's place, which shows it on the face),
     the map's class on the right-or-wrong face, a two-map sheet's pair once
     unblinded, Show (the row's pictures, under the head "Pictures"; Hide on
     the row whose pictures are open, since a press there closes them) or,
     on a sheet with places and no pictures, Street map (r5-e H4: a sheet
     with pictures says Street map once, in the open pane's head), and the
     answer. */
  function renderList(focusKey) {
    const current = focusedControl();
    const was = focusKey ? { key: focusKey, place: current ? current.place : -1 } : current;
    const unblinded = !state.blind && state.revealed;
    const showMapClass = wrongFace;
    const showPair = twoMap && unblinded;
    const rows = visibleRows();
    // The pictures column: Show (where this sheet can show pictures) or
    // Street map (where it has places and no pictures); none when it has
    // neither (finding 10).
    const placeColumn = picturesPossible || Boolean(data.has_place);
    const head = OE.el('tr', {},
      OE.el('th', {}, '#'),
      showMapClass ? OE.el('th', {}, 'Map class') : null,
      showPair ? OE.el('th', {}, 'Map A / Map B') : null,
      // WORDING: r4h A/F21 (the column of a sheet with no pictures holds only Street map)
      placeColumn ? OE.el('th', {}, picturesPossible ? 'Pictures' : W.streetMap) : null,
      OE.el('th', {}, wrongFace ? 'Is the map right?' : 'Class seen'));
    const lines = rows.map((w) => {
      const h = w.handle;
      const k = place(h);
      const extra = state.extra.get(h) || {};
      const pair = Array.isArray(extra.pair) ? extra.pair : ['', ''];
      return OE.el('tr', { 'data-handle': h },
        OE.el('td', {}, OE.el('button', {
          class: 'link', 'data-focus': 'n:' + h, 'aria-label': W.openRow(k), onclick: () => setView('face', h),
        }, String(k))),
        showMapClass ? OE.el('td', {}, extra.mapClass || '') : null,
        showPair ? OE.el('td', {}, (pair[0] || '') + ' / ' + (pair[1] || '')) : null,
        placeColumn ? OE.el('td', {}, OE.el('span', { class: 'row', style: 'gap:2px' },
          // WORDING: r5e H4 (the column's head says "Pictures" once; the button
          // says what its press does, Show, or Hide on the row whose pictures
          // are open, and its name starts with that word; markPictures keeps
          // both in step while the rows stand)
          picturesPossible ? OE.el('button', {
            class: 'link', 'data-focus': 'i:' + h, 'aria-expanded': String(state.shown === h),
            'aria-controls': previewId,
            'aria-label': (state.shown === h ? W.hidePictures : W.showPictures)(k), onclick: () => togglePictures(h),
          }, state.shown === h ? 'Hide' : 'Show') : null,
          // Street map on each row only where the sheet shows no pictures; else
          // once, in the open pane's head (r5-e H4).
          data.has_place && !picturesPossible ? OE.el('button', {
            class: 'link', 'data-focus': 'm:' + h, 'aria-disabled': String(linking.has(h)),
            'aria-label': W.openMapOf(k), onclick: () => openMap(h),
          }, W.streetMap) : null)) : null,
        OE.el('td', {}, labelCell(w)));
    });
    body.replaceChildren(...(lines.length ? lines
      : [OE.el('tr', {}, OE.el('td', { colspan: '99', class: 'faint' }, 'No rows'))]));
    table.querySelector('thead').replaceChildren(head);
    updateProgress();
    restoreFocus(was);
  }

  /* A window's map link, asked of the server at the press (the page copy
     holds no place), then opened in the user's browser. Busy while asked
     (aria-disabled), so a second press asks nothing. */
  async function openMap(h) {
    if (linking.has(h)) return;
    const mark = (busy) => {
      const b = wrap.querySelector('[data-focus="' + CSS.escape('m:' + h) + '"]');
      if (b) b.setAttribute('aria-disabled', String(busy));
    };
    linking.add(h);
    mark(true);
    try {
      const got = await OE.call(app, 'olmoearth_app_window_link', { labels_csv_path: data.path, sheet_id: data.sheet_id, handle: h });
      const url = got && typeof got.url === 'string' ? got.url : '';
      if (url.startsWith('https://www.openstreetmap.org/')) await OE.open(app, url);
      else OE.toast(W.noLink(place(h)), true);
    } catch (err) {
      const said = bare(err.message || 'the link could not be made');
      const text = 'The map was not opened: ' + said + '.';
      if (said.startsWith(OE.CHANGED_UNDER_PAGE)) { markStale(); reopenToast(text); } else OE.toast(text, true);
    } finally {
      linking.delete(h);
      mark(false);
    }
  }

  /* The Pictures buttons say which window's pane is open, in place: the rows
     are not rebuilt, so nothing moves the focus or drops a typed class. A
     row's button in the list also says what its press does now, in its word
     and at the start of its name (r5-e H4, after review: Hide on the row
     whose pictures are open, Show on every other); the face's button, which
     has the same focus key, keeps "Pictures" (body.contains). */
  function markPictures() {
    for (const b of wrap.querySelectorAll('[data-focus^="i:"]')) {
      const open = b.dataset.focus === 'i:' + state.shown;
      b.setAttribute('aria-expanded', String(open));
      if (!body.contains(b)) continue;
      // WORDING: r5e H4
      b.textContent = open ? 'Hide' : 'Show';
      b.setAttribute('aria-label', (open ? W.hidePictures : W.showPictures)(place(b.dataset.focus.slice(2))));
    }
  }

  /* Close the pane; the focus goes back to the window's Pictures button (the
     user pressed a button in the pane, so the focus was in the panel). */
  function closePictures(h) {
    state.shown = null;
    state.test = null;
    renderPreview();
    markPictures();
    OE.focusIfHere(wrap.querySelector('[data-focus="' + CSS.escape('i:' + h) + '"]'));
  }

  /* A window's Pictures button: opens its pane, or closes it when it is open.
     Its press brings the question into view (reveal, r5-f F01). */
  function togglePictures(h) {
    if (state.shown === h) { closePictures(h); return; }
    loadPictures(h, true);
  }

  /* The pictures of a window, asked of the server by its handle (it finds
     the window from the files on disk). An answer is kept until pictures
     are switched on or off (imagesRun); a failure is not kept, so the next
     press asks again. The call waits OE.PICTURES_WAIT_MS (r5-a L1: as long
     as the server can take). A call that throws shows one fixed sentence
     (OE.picturesFailed), never the thrown error's text. A failure stays in
     the pane while its window is shown (state.failed), with Try again where
     another try can get past it. After a press of Pictures (Show in the
     list: reveal), an answer that is the question is brought into view
     (questionInView, r5-f F01), also when it was kept from an earlier press;
     a pane that follows an answer, Previous or Next leaves the scroll as it
     is (r5-k K4: an answer that moves on and an arrow key bring the new
     square's heading into view first, headInView; a press of Next or
     Previous, in that heading's row, moves nothing). */
  async function loadPictures(h, reveal = false) {
    state.shown = h;
    state.test = null;
    markPictures();
    if (state.images.has(h)) { renderPreview(); if (reveal) questionInView(state.images.get(h)); return; }
    const run = state.imagesRun;
    state.failed = null;
    state.images.set(h, { loading: true });
    renderPreview();
    let answer;
    try {
      answer = await OE.call(app, 'olmoearth_app_window_image',
        { labels_csv_path: data.path, sheet_id: data.sheet_id, handle: h }, { timeout: OE.PICTURES_WAIT_MS });
    } catch (_) {
      answer = OE.picturesFailed();
    }
    if (run !== state.imagesRun) return;  // switched on or off meanwhile: asked again then
    // The question means pictures are off; pictures mean they are on.
    if (answer && answer.needs_consent) state.picturesOn = false;
    else if (answer && answer.ok) state.picturesOn = true;
    if (answer && (answer.ok || answer.needs_consent)) state.images.set(h, answer);
    else {
      state.images.delete(h);
      if (state.shown === h) state.failed = { h, answer: answer || {} };
    }
    if (state.shown === h) { renderPreview(answer); if (reveal) questionInView(answer); }
  }

  /* The question that needs a yes, brought into view (r5-f F01) after a
     press of Pictures (Show in the list), and of One square while its pane
     is open with the question kept (setView). It was drawn under the answers
     (above them on the face since r5-k K4; under the table in the list);
     on a panel of fixed height (full screen, a host that caps the inline
     height) only its first line showed above the pinned footer and "Turn on
     pictures" was out of view, so a square could be answered with nothing
     looked at. The pane scrolls the least that shows it whole above the
     footer when it fits (its scroll margin: the footer's height and 8 px);
     a pane taller than the panel shows its top, and then the least that
     brings the row of "Turn on pictures" (data-ask-yes) into view above the
     footer. A pane already in view does not move; a pane that follows an
     answer, Previous or Next scrolls nothing (r5-k K4: the answer that moves
     on, or an arrow key, has already brought the new square's heading into
     view, headInView; a press of Next or Previous, in that heading's row,
     moves nothing). The focus stays where it is:
     the pane is a polite live region, so a screen reader hears the
     question. No word changes. The scroll waits two frames for the layout
     to settle (after the check): run at once, it could land while the face
     was redrawn shorter, and the browser then held it short of the row. */
  function questionInView(answer) {
    if (!answer || !answer.needs_consent || typeof preview.scrollIntoView !== 'function') return;
    const later = typeof requestAnimationFrame === 'function' ? requestAnimationFrame : (fn) => setTimeout(fn, 16);
    later(() => later(() => { if (preview.isConnected) bringQuestionUp(); }));
  }
  function bringQuestionUp() {
    const foot = wrap.querySelector('.sheet-footer');
    const margin = ((foot ? foot.offsetHeight : 0) + 8) + 'px';
    preview.style.scrollMarginBottom = margin;
    preview.scrollIntoView({ block: 'nearest' });
    // Then the least scroll of the page that puts the row of "Turn on pictures" above the footer, worked out from
    // the two rectangles (after the check: scrollIntoView with a scroll margin stopped short of it here).
    const yes = preview.querySelector('[data-ask-yes]');
    if (!yes || !foot) return;
    const at = yes.getBoundingClientRect();
    const below = at.bottom + 8 - foot.getBoundingClientRect().top;
    if (below > 0) window.scrollBy(0, Math.min(below, Math.max(0, at.top)));
    else if (at.top < 0) window.scrollBy(0, at.top - 8);
  }

  /* Try again in the pane (r5-a L1): the same call once more for this
     window. The pressed button goes with the old pane, so the focus goes to
     the pane's heading, then and once the answer is drawn. */
  async function retryPictures(h) {
    const asked = loadPictures(h);
    focusPaneHeading();
    await asked;
    focusPaneHeading();
  }

  /* Pictures switched on or off by the user's own press, after the question. */
  async function setPictures(on) {
    try {
      await OE.call(app, 'olmoearth_app_imagery', { on });
    } catch (err) {
      OE.toast(err.message, true);
      return;
    }
    state.picturesOn = Boolean(on);
    state.images.clear();
    state.failed = null;
    state.imagesRun += 1;
    state.test = null;
    const h = state.shown;
    if (on && h !== null) {
      await loadPictures(h);
      // The pressed button went with the old pane: the focus goes to the pane's heading.
      const lost = !document.activeElement || document.activeElement === document.body;
      const heading = preview.querySelector('h2, h3');
      if (lost && heading) { heading.setAttribute('tabindex', '-1'); OE.focusIfHere(heading); }
    }
    else if (h !== null) closePictures(h);
  }

  /* The pane: the question until pictures are on, then the rule, the
     pictures with their caption, and what a label then is. In the list it
     is headed by its window; on the face, whose heading names the window,
     by "Pictures". ``shown`` overrides the cache (an answer not kept, such
     as a failure). */
  function renderPreview(shown) {
    const h = state.shown;
    work.classList.toggle('pane', h !== null);  // beside the table only while open
    if (h === null) {
      // No pane: while pictures are off, the one line that says how to see
      // them (A/sheet-1), under the answers; added once, so a live region
      // does not say it again on every answer. A sheet that can show no
      // pictures has no such line (finding 10). In the list the line names
      // the press there, a row's Show (r5-e H4, after review: no button of
      // the list reads "Pictures"); its words change only when the view
      // does, so the live region says it once per switch.
      if (state.picturesOn || !picturesOff) preview.replaceChildren();
      else {
        const off = state.view === 'list' ? W.picturesOffList : W.picturesOff;
        if (picturesOff.textContent !== off) picturesOff.textContent = off;
        if (preview.childNodes.length !== 1 || preview.firstChild !== picturesOff) preview.replaceChildren(picturesOff);
      }
      return;
    }
    if (state.test) { renderTest(); return; }
    const k = place(h);
    // A failure is kept only while its window stays shown (r5-a L1): another
    // window's pane drops it, and coming back asks again, as before.
    if (state.failed && state.failed.h !== h) state.failed = null;
    const answer = shown || state.images.get(h) || (state.failed && state.failed.answer) || { loading: true };
    // In the list, the head holds the square's one Street map, where the
    // sheet has places (r5-e H4: no longer on every row); the face has its
    // own, beside Pictures.
    const head = OE.el('div', { class: 'spread' },
      state.view === 'face' ? OE.el('h3', {}, 'Pictures') : OE.el('h2', {}, W.paneHead(k)),
      OE.el('span', { class: 'row', style: 'gap:4px' },
        state.view === 'list' && data.has_place ? OE.el('button', { class: 'link', 'data-focus': 'm:' + h, 'aria-disabled': String(linking.has(h)), 'aria-label': W.openMapOf(k), onclick: () => openMap(h) }, W.streetMap) : null,
        OE.el('button', { class: 'link', onclick: () => closePictures(h) }, 'Close pictures')));
    // While the next window's pictures load, the pane keeps the height it had (r5-f F01, after the check): a short
    // "Loading" line made the page shorter, the browser squeezed the scroll, and the face jumped when Next was
    // pressed with the pane open. Any other answer drops the kept height.
    if (answer.loading) {
      if (preview.childNodes.length && !preview.style.minHeight) preview.style.minHeight = preview.offsetHeight + 'px';
      preview.replaceChildren(head, OE.el('div', { class: 'faint' }, 'Loading the pictures...'));
      return;
    }
    preview.style.minHeight = '';
    if (answer.needs_consent) {
      // r5-k K7: the question's three short lines (OE.askLines), "Turn on pictures", the closed fold "What is
      // sent, in full" with the two hosts and the test scene's question (OE.askFold), then the test scene's button:
      // the press comes after the fold that says what it sends.
      preview.replaceChildren(head, OE.el('div', { class: 'card summary stack oe-ask' },
        OE.askLines(answer),
        // The row questionInView keeps in view on a short panel (r5-f F01).
        OE.el('div', { class: 'row', style: 'justify-content:flex-end', 'data-ask-yes': '' },
          OE.el('button', { onclick: () => setPictures(true) }, 'Turn on pictures')),
        OE.askFold(answer),
        answer.test_ask ? OE.el('div', { class: 'row', style: 'justify-content:flex-end' },
          OE.el('button', { class: 'link', onclick: () => loadTest() }, 'Try the public test scene')) : null));
      return;
    }
    if (!answer.ok) {
      // Try again (r5-a L1) under a failure another try can get past: the
      // server's again (a slow fetch, the catalogue, the image files), or a
      // call that threw (OE.picturesFailed). It repeats the call for this
      // window. Any other refusal would be the same again: its sentence alone.
      preview.replaceChildren(...[head, OE.el('div', { class: 'error small' }, answer.error || 'No pictures.'),
        answer.again ? OE.el('div', { class: 'row' },
          OE.el('button', { type: 'button', class: 'small', 'data-focus': 'again', onclick: () => retryPictures(h) },
            OE.TRY_AGAIN)) : null].filter(Boolean));
      return;
    }
    if (answer.no_image) {
      // No month of the map's period has a scene with data on this window
      // (r5-a L2): one fixed line in place of a list of empty months, and,
      // on a face that takes Cannot tell, what to press. The pane keeps its
      // test scene and its switch.
      preview.replaceChildren(head,
        OE.el('div', { class: 'small oe-no-image' }, W.noImage + (takesQ ? OE.NO_IMAGE_PRESS : '')),
        OE.el('div', { class: 'row', style: 'justify-content:flex-end;gap:16px' },
          OE.el('button', { class: 'link', onclick: () => loadTest() }, 'Public test scene'),
          OE.el('button', { class: 'link', onclick: () => setPictures(false) }, 'Turn off pictures')));
      return;
    }
    const pictures = (answer.pictures || []).filter(goodPicture);
    // The rule shows above the answers from the first open (ruleBlock,
    // A/sheet-2); here only when the page copy had none and this answer
    // has one: its Present half, then its Absent half word for word, then,
    // on a face that takes Cannot tell (not a certificate plan's), that a
    // square with no clear date is Cannot tell (r4-m M8: so a square hidden
    // by cloud is not pressed Absent by the rule's own words). Both rules
    // come from imagery.rule_for, so this is the fallback for a page copy
    // whose rule could not be read; the rule block says the cloud line
    // otherwise. Its scope is not repeated here: the Check card and the
    // result show it. That no rule was set is said on
    // the face, in the rule's place, from the first open (noRuleLine, r5-b
    // J26), and no longer here; a rule these pictures bring takes that
    // line's place.
    let ruleLines = null;
    if (!pageRule && answer.reading_rule) {
      const [present, absent] = ruleHalves(answer.reading_rule);
      ruleLines = OE.el('div', {}, OE.el('div', { class: 'small', style: 'font-weight:600' }, 'Rule: ' + present),
        absent ? OE.el('div', { class: 'small', style: 'font-weight:600' }, absent) : null,
        takesQ ? OE.el('div', { class: 'small' }, W.cloudLine) : null);
      if (noRuleLine) noRuleLine.hidden = true;
    }
    const dated = Array.isArray(answer.dates) && answer.dates.length > 0;
    const clouds = dated ? cloudFold(answer.dates) : null;
    preview.replaceChildren(...[head, ruleLines].filter(Boolean),
      dated
        ? datesBlock(answer.dates, W.label(k))
        : OE.el('div', { class: 'row', style: 'align-items:flex-start;gap:12px' },
          pictures.map((p) => OE.el('figure', { style: 'margin:0' },
            OE.el('img', { src: p.src, width: p.width, height: p.height, alt: p.label + ', ' + W.label(k), style: 'image-rendering:pixelated;display:block;border:1px solid var(--rule);border-radius:4px' }),
            OE.el('figcaption', { class: 'small' }, p.label)))),
      ...[clouds].filter(Boolean),
      OE.el('div', { class: 'footnote' }, answer.caption),
      OE.el('div', { class: 'footnote' }, answer.label_is),
      OE.el('div', { class: 'row', style: 'justify-content:flex-end;gap:16px' },
        OE.el('button', { class: 'link', onclick: () => loadTest() }, 'Public test scene'),
        OE.el('button', { class: 'link', onclick: () => setPictures(false) }, 'Turn off pictures')));
  }

  /* A picture the server sent: a data URL and whole-number sizes. */
  function goodPicture(p) {
    return OE.goodPicture(p);
  }

  /* One block per date, the chosen rendering only, with the colour switch
     (OE.datesBlock). On the face the dates sit side by side inline, wrapped
     onto as many rows as they need (r5-k K4: no longer a strip that scrolls
     sideways), and are a grid in full screen (.face-dates in common.js); in
     the list, one under another. Each date is headed
     by its day alone (dateHead): its scene cloud is in the pane's fold
     (cloudFold, r4-m M8). */
  function datesBlock(dates, what) {
    return OE.datesBlock(dates, what, {
      colour: state.colour,
      strip: state.view === 'face',
      heading: dateHead,
      unit: W.one,
      choose: (kind) => {
        state.colour = kind;
        renderPreview();
        OE.focusIfHere(preview.querySelector('[data-focus="colour:' + kind + '"]'));
      },
    });
  }

  /* A date's label as the server writes it ("10 Apr 2024, scene cloud 2%",
     imagery._date_label): its day, and its scene cloud. "2% scene cloud" on
     a date's heading was read as a clear square, but it is the whole
     satellite image's (reader test 3, r4-m M8): the heading keeps the day
     (dateHead), and the label whole goes to the pane's closed fold
     (cloudFold), after the fold's first line, which says whose cloud it is.
     A label with no scene cloud (a month with no scene) is its own heading
     and is not in the fold; no such date, no fold. Declared as functions, so
     a pane drawn before this point of the editor finds them. */
  function sceneCloud(label) {
    return /^(.+?), (scene cloud .+)$/.exec(label);
  }
  function dateHead(label) {
    const m = sceneCloud(label);
    return m ? m[1] : label;
  }
  /* r5-p: after each date's label, its cloud mask line as the server words
     it (the date's mask_line, imagery.mask_line: none on the square, cloud or
     shadow on a share of it, or no mask for the date), set in a little
     (.oe-mask-line). The labels stay word for word; an answer with no such
     line (an older server) draws the fold as before. */
  function cloudFold(dates) {
    const said = dates.filter((d) => d && sceneCloud(String(d.label || '')) !== null);
    if (!said.length) return null;
    // WORDING: r4m M8 (W.cloudFold, W.cloudWhole)
    return OE.el('details', { class: 'oe-clouds small' }, OE.el('summary', {}, W.cloudFold),
      OE.el('div', {}, W.cloudWhole), ...said.map((d) => [OE.el('div', {}, String(d.label)),
        typeof d.mask_line === 'string' && d.mask_line
          ? OE.el('div', { class: 'oe-mask-line', style: 'padding-left:12px' }, d.mask_line) : null]));
  }

  /* A rule's two halves (r4-m M8), each in the rule's own words: up to its
     "Absent:" and from it on; a rule with no "Absent:" after its start is
     one half, the second null. */
  function ruleHalves(rule) {
    const text = String(rule);
    const at = text.search(/\bAbsent\s*:/i);
    return at > 0 ? [text.slice(0, at).trim(), text.slice(at).trim()] : [text, null];
  }

  /* The public test scene: the user's press, on the button after the fold
     that holds its question, what it sends (r5-k K7). Shown in the pane in
     place of the window; nothing is labelled. */
  async function loadTest() {
    const from = state.shown;
    state.test = { from, answer: { loading: true } };
    renderPreview();
    focusPaneHeading();  // the pressed button went with the old pane
    let answer;
    try {
      // As a window's pictures (r5-a L1): the long wait, and a call that
      // throws shows the fixed sentence, never the thrown error's text.
      answer = await OE.call(app, 'olmoearth_app_window_image', { test_scene: true },
        { timeout: OE.PICTURES_WAIT_MS });
    } catch (_) {
      answer = OE.picturesFailed();
    }
    if (!state.test || state.test.from !== from) return;  // closed meanwhile
    state.test.answer = answer || { ok: false, error: 'No answer.' };
    renderPreview();
    focusPaneHeading();
  }

  /* After the pane is rebuilt: the focus goes to its heading only when it was
     in the pane or lost, never away from a row the user went back to. */
  function focusPaneHeading() {
    const active = document.activeElement;
    const lost = !active || active === document.body || preview.contains(active);
    const heading = preview.querySelector('h2, h3');
    if (lost && heading) { heading.setAttribute('tabindex', '-1'); OE.focusIfHere(heading); }
  }

  function closeTest() {
    state.test = null;
    renderPreview();
    focusPaneHeading();
  }

  function renderTest() {
    const answer = state.test.answer;
    const head = OE.el('div', { class: 'spread' },
      state.view === 'face' ? OE.el('h3', {}, 'Public test scene') : OE.el('h2', {}, 'Public test scene'),
      OE.el('button', { class: 'link', onclick: closeTest }, W.back(place(state.test.from))));
    if (answer.loading) { preview.replaceChildren(head, OE.el('div', { class: 'faint' }, 'Loading the test scene...')); return; }
    if (!answer.ok) {
      // Try again as under a window's pictures (r5-a L1): the test scene's call once more.
      preview.replaceChildren(...[head, OE.el('div', { class: 'error small' }, answer.error || 'No pictures.'),
        answer.again ? OE.el('div', { class: 'row' },
          OE.el('button', { type: 'button', class: 'small', 'data-focus': 'again', onclick: () => loadTest() },
            OE.TRY_AGAIN)) : null].filter(Boolean));
      return;
    }
    // Its dates headed by their day too, their scene cloud in the fold (r4-m M8).
    const dates = Array.isArray(answer.dates) ? answer.dates : [];
    preview.replaceChildren(head,
      OE.el('div', { class: 'small faint' }, W.notOfSheet),
      datesBlock(dates, 'the public test scene'),
      ...[cloudFold(dates)].filter(Boolean),
      OE.el('div', { class: 'footnote' }, answer.caption));
  }

  /* A save 1.5 s after the last answer; a run of answers under 1.5 s apart
     saves at the latest 5 s after its first unsaved answer, so a long run of
     keys is not held only in the page (at least 200 ms between tries). */
  const SAVE_QUIET_MS = 1500;
  const SAVE_MAX_WAIT_MS = 5000;
  let timer = null;
  let firstUnsaved = 0;
  function schedule() {
    clearTimeout(timer);
    if (state.stale) return;  // a save would be refused again: the sheet must be reopened
    const now = Date.now();
    if (!firstUnsaved) firstUnsaved = now;
    const wait = Math.max(200, Math.min(SAVE_QUIET_MS, firstUnsaved + SAVE_MAX_WAIT_MS - now));
    timer = setTimeout(() => save({ auto: true }), wait);
  }

  /* Nothing saves any more: the sheet changed under the page, or (why
     'read', r5-b J15) this panel removed a set of answers and could not read
     the sheet again, so the footer does not say it changed elsewhere. */
  function markStale(why) {
    state.stale = true;
    state.staleWhy = why || 'changed';
    clearTimeout(timer);
    updateProgress();
    tellClaude();
  }

  /* A toast about the sheet that changed under the page (each toast after
     markStale, and save's on a press of Save then): the last one is kept,
     so that oeLeave clears it while it is still the one shown (r4-l L3).
     Save's toast that the changes not saved here are not kept is not one:
     it is about an unsaved change, and stays. Each is an error but the
     clear's whose sheet could not be read again (isError false, r5-i B7):
     that the answers went is no error, so it is said politely; the footer
     says what failed, as Reopen the sheet's description. */
  function reopenToast(text, isError = true) {
    state.reopenSaid = text;
    OE.toast(text, isError);
  }

  /* r5-k K5: a status of this sheet's saves, in its own place in the footer
     (slot: saveSaid under the row of the finish button and Save labels, or
     countSaid beside the counter), in its words, no longer in the status
     line at the page's foot: a plain message polite (role status), an error
     at once (role alert, assertive, in the error colour), as OE.toast says
     them. Once this editor is off the page (another sheet opened in its
     place), it has no footer to show it: the status line says it, as before. */
  function footerSay(slot, text, isError = false) {
    if (!wrap.isConnected) { OE.toast(text, isError); return; }
    slot.setAttribute('role', isError ? 'alert' : 'status');
    slot.setAttribute('aria-live', isError ? 'assertive' : 'polite');
    slot.classList.toggle('error', isError);
    slot.classList.toggle('muted', !isError);
    slot.textContent = text;
    fitFoot();
  }

  /* r5-k K5: a save that went through, said beside the counter in the place
     of "Saved automatically" (savedNote, a polite status shown only while
     nothing waits: updateProgress), never under Save labels or the finish
     button, and "Saved automatically" is not said a second time; the save's
     own line under Save labels is emptied, so an earlier "Not saved: ..."
     goes. Once this editor is off the page, the status line says it, as
     before. */
  function counterSay(text) {
    if (!wrap.isConnected) { OE.toast(text); return; }
    saveSaid.textContent = '';
    savedNote.textContent = text;
    fitFoot();
  }

  /* r5-k K5 (r5-j B8, WCAG 2.4.11): the sticky footer's height, kept on the
     sheet as --oe-foot, from which each of its controls takes its scroll
     margin (common.js: that height and 8 px, as bringQuestionUp keeps the
     pictures' question), so a control the focus brings into view ends above
     the footer whatever the footer holds now (its row wrapped on a narrow
     panel, Save labels, a line of the saves, the finish press's slot). Run
     where the footer changes before a focus move in the same task
     (updateProgress, footerSay, counterSay, the finish press) and by a
     ResizeObserver for a width that changes (below); a footer not drawn
     (offsetHeight 0: the editor not on the page yet) changes nothing. */
  function fitFoot() {
    const foot = wrap.querySelector(':scope > .sheet-footer');
    const tall = foot ? foot.offsetHeight : 0;
    if (tall) wrap.style.setProperty('--oe-foot', tall + 'px');
  }

  /* "Square 4", "Squares 4 and 9", "Squares 2, 4 and 9" (the sheet's unit
     word, W.places): windows by their place. */
  function windowsWord(places) {
    return W.places([...places].sort((a, b) => a - b).map(String));
  }

  /* What a save that failed asks of the user: a press of Save labels while
     this editor is on the page; once another sheet was opened in its place
     (its Save is gone), to open this sheet again, with the windows not saved
     named by their place. */
  function tryAgain() {
    if (wrap.isConnected) return 'Press Save labels to try again.';
    const places = [...state.dirty].map(place).filter(Boolean);
    return (places.length ? windowsWord(places) + ' of ' + data.name + (places.length === 1 ? ' is' : ' are')
      + ' not saved: open that sheet again to label ' + (places.length === 1 ? 'it' : 'them') + '.'
      : 'Open ' + data.name + ' again to go on.');
  }

  /* The sheet as the server holds it now, on this page's face, to take in a
     save made elsewhere; null when it cannot be read. A sheet drawn again,
     or whose rule or face changed, cannot be taken in: the page stops
     saving and offers to reopen it. */
  async function readFresh() {
    try {
      return await OE.call(app, 'olmoearth_app_sheet', { labels_csv_path: data.path, face });
    } catch (err) {
      const said = bare(err.message);
      if (said.startsWith(OE.CHANGED_UNDER_PAGE) || said.startsWith(OE.FACE_WRONG_REFUSED)
          || said.startsWith(OE.FACE_CLASS_REFUSED)) {
        markStale();
        reopenToast('This sheet changed since this panel opened it. Reopen it to go on.');
      } else {
        OE.toast('The sheet could not be read again: ' + said + '.', true);
      }
      return null;
    }
  }

  /* Take in a fresh read of the sheet (a save made elsewhere): a window not
     changed here takes the label saved there; a window changed here keeps
     its change where the label there is still state.base, and takes the
     label saved there where both changed it (the first save wins). The
     windows shown, their order and the focus stay. True when the page now
     shows something it did not show before (a label, or that the sheet must
     be reopened); the toast then says so, naming by their place the windows
     whose change here gave way. A class being typed keeps its text and its
     caret across the redraw. mine: the read follows this page's own save,
     whose answer says the server worked out more windows (rederived): every
     change made here wins, and the toast says the sheet was read again.
     changes counts each take-in that showed a change (finishPress). */
  let changes = 0;
  function adopt(fresh, { mine = false } = {}) {
    lastCheck = Date.now();
    const list = fresh && Array.isArray(fresh.windows) ? fresh.windows : [];
    const got = new Map(list.filter((w) => w && typeof w.handle === 'string')
      .map((w) => [w.handle, String(w.answer ?? '').trim()]));
    const same = fresh && fresh.sheet_id === data.sheet_id && (fresh.label_rule || null) === rule
      && fresh.face === face && got.size === n && windows.every((w) => got.has(w.handle));
    if (!same) {
      markStale();
      reopenToast('This sheet changed since this panel opened it. Reopen it to go on.');
      changes += 1;
      return true;
    }
    const gaveWay = [];
    let differed = false;
    for (const w of windows) {
      const h = w.handle;
      const there = got.get(h);
      const shown = state.labels.get(h) || '';
      if (state.dirty.has(h) && (mine || there === state.base.get(h))) {
        // Changed here only (or after this page's own save): the change stays, to be sent.
      } else {
        if (there !== shown) {
          differed = true;
          if (state.dirty.has(h)) gaveWay.push(place(h));
        }
        state.labels.set(h, there);
        state.dirty.delete(h);
      }
      state.base.set(h, there);
    }
    // r5-k K2: an answer changed elsewhere and taken in undoes the finish press too (setLabel's rule), before the
    // redraw below locks a two-map sheet's switch again.
    if (differed && !mine) {
      resultSent = false;
      resultShown = false;
    }
    if (typeof fresh.version === 'string') data.version = fresh.version;
    state.mix = fresh.mix_problem
      ? { problem: String(fresh.mix_problem), column: String(fresh.clear_column || '') } : null;
    keptNote = gaveWay.length
      ? windowsWord(gaveWay) + (gaveWay.length === 1 ? ' keeps the label' : ' keep the labels') + ' saved there.' : '';
    // Drawn again where it was: the window shown, the list's order, the focused
    // control, and a class being typed (its text and caret).
    const active = document.activeElement;
    const key = active && wrap.contains(active) && active.dataset ? active.dataset.focus || null : null;
    const typedIn = () => (state.other !== null
      ? wrap.querySelector('input[data-focus="' + CSS.escape('o:' + state.other) + '"]') : null);
    const field = typedIn();
    const draft = field ? { value: field.value, start: field.selectionStart, end: field.selectionEnd } : null;
    render(key || undefined);
    const again = draft ? typedIn() : null;
    if (again) {
      again.value = draft.value;
      try { again.setSelectionRange(draft.start, draft.end); } catch (_) { /* a field with no caret */ }
    }
    if (differed) {
      // r5-k K5: after this page's own save, beside the counter as the save's
      // own words ("Saved to ...", counterSay); a save made elsewhere, beside
      // the counter it changed (3g: the count that dropped had its reason far
      // below it).
      if (mine) counterSay(W.rereadMine(data.name));
      else footerSay(countSaid, 'This sheet was saved elsewhere; it shows those labels now.' + (keptNote ? ' ' + keptNote : ''));
      changes += 1;
    }
    tellClaude();
    return differed;
  }

  /* A save refused as stale: read the sheet, take it in, and send what is
     left with the new version; at most STALE_RESENDS times in a row. */
  async function resync() {
    state.syncing = true;
    clearTimeout(timer);
    let fresh = null;
    try {
      fresh = await readFresh();
      if (fresh) adopt(fresh);
    } finally {
      state.syncing = false;
    }
    if (!fresh || state.stale) { updateProgress(); return; }
    if (!pending()) { keptNote = ''; updateProgress(); return; }
    if (staleResends >= STALE_RESENDS) {
      // r5-k K5: beside Save labels, in the footer (3f: it was under the card).
      footerSay(saveSaid, 'Not saved: the sheet keeps changing elsewhere. ' + tryAgain(), true);
      updateProgress();
      return;
    }
    staleResends += 1;
    await save();
  }

  /* Whether the sheet was saved elsewhere since this page read it: the
     server's version (olmoearth_app_sheet_progress, counts only) against
     data.version; if so the sheet is read and taken in (adopt). At most once
     every CHECK_EVERY_MS unless forced (the finish press), and never while a
     save or a read is under way, a class is being typed, the page is not
     shown, or it stopped saving; a check already under way is waited for.
     True when the page now shows a change, or when the version moved on and
     the sheet could not be read again (the page has not seen how it
     changed: a finish press then sends nothing). r5-k K3: the counts also
     say whether the sheet gave its result (graded); one that did since this
     editor was drawn (Claude graded it after the finish press, the version
     unchanged) is read and drawn again (replaceEditor), its answers locked,
     once nothing here waits to be saved (else it is taken in, and the next
     check, after the save, draws it again). */
  let checking = null;
  function check({ force = false } = {}) {
    if (checking) return checking;
    if (!wrap.isConnected || state.saving || state.syncing || state.stale || reopening) return Promise.resolve(false);
    if (typeof data.version !== 'string') return Promise.resolve(false);
    if (!force && (Date.now() - lastCheck < CHECK_EVERY_MS || state.other !== null)) return Promise.resolve(false);
    lastCheck = Date.now();
    checking = (async () => {
      let got = null;
      try {
        got = await OE.call(app, 'olmoearth_app_sheet_progress', { labels_csv_path: data.path });
      } catch (_) {
        return false;  // no word from the server: nothing to take in
      }
      const graded = Boolean(got && got.graded && typeof got.graded === 'object') && !gaveResult;
      if (!got || typeof got.version !== 'string' || (got.version === data.version && !graded)) return false;
      if (state.saving || state.syncing || state.stale) return false;
      state.syncing = true;
      try {
        const fresh = await readFresh();
        if (graded && fresh && fresh.graded && typeof fresh.graded === 'object' && !pending() && wrap.isConnected) {
          replaceEditor(fresh);
          return true;
        }
        const changed = fresh ? adopt(fresh) : true;
        if (!pending()) keptNote = '';  // said by adopt's toast; no save to carry it
        return changed;
      } finally {
        state.syncing = false;
        updateProgress();
      }
    })().finally(() => { checking = null; });
    return checking;
  }

  /* Draw a new editor in place of this one from a fresh read; its heading
     takes the focus if this one had it. The changes not saved here are not
     kept (every caller saves them first, or the sheet changed under it). */
  function replaceEditor(fresh, said) {
    clearTimeout(timer);
    wrap.oeDispose();
    const hadFocus = wrap.contains(document.activeElement);
    const editor = OE.sheetEditor(app, fresh, { headerAction, onSaved });
    wrap.replaceWith(editor);
    if (said) OE.toast(said);
    const heading = editor.querySelector('h1');
    if (heading && hadFocus) { heading.setAttribute('tabindex', '-1'); OE.focusIfHere(heading); }
  }

  /* Read the sheet again and draw a new editor in place of this one (after a
     save was refused because the sheet changed under the page), on the face
     the server offers first. */
  let reopening = false;
  async function reopen() {
    if (reopening) return;
    reopening = true;
    reopenButton.setAttribute('aria-disabled', 'true');
    try {
      const fresh = await OE.call(app, 'olmoearth_app_sheet', { labels_csv_path: data.path });
      replaceEditor(fresh, 'The sheet was read again: ' + data.name + '.');
    } catch (err) {
      OE.toast('The sheet could not be read: ' + bare(err.message) + '.', true);
      reopening = false;
      reopenButton.setAttribute('aria-disabled', 'false');
    }
  }

  /* Save what is waiting, after any save or read in flight; true when nothing is left. */
  async function settle() {
    while (state.saving || state.syncing) await wait(100);
    if (pending()) await save();
    return !pending() && !state.stale;
  }

  /* Remove one of a sheet's two sets of answers: clear a label column on
     every window (the user's press on the button beside the notice of a
     mix, r5-b J21), then read the sheet on face ``to``. The clear names the
     version this page holds: a sheet saved elsewhere since is read and
     taken in first, and the user presses the button again. No
     answer is taken meanwhile (busyArea). A clear the server took whose
     sheet then cannot be read says the answers were removed, and the page
     offers to reopen the sheet (markStale: this copy is no longer the
     file's). The toasts name the set by what it holds (answersOf), never by
     its column. Nothing is cleared while the answers are locked (r5-k K3: a
     sheet that gave its result, or one not saving). */
  async function clearAndOpen(column, to) {
    if (state.clearing || locked()) return;
    state.clearing = true;
    busyArea(true);
    let stale = false;
    try {
      if (!(await settle())) return;
      const args = { labels_csv_path: data.path, sheet_id: data.sheet_id, face, rule, clear_column: column };
      if (typeof data.version === 'string') args.version = data.version;
      let cleared = false;
      try {
        await OE.call(app, 'olmoearth_app_save_labels', args);
        cleared = true;
      } catch (err) {
        const said = bare(err.message || 'the column was not cleared');
        if (err.stale) stale = true;
        else {
          const text = 'Not cleared: ' + said + '.';
          if (said.startsWith(OE.CHANGED_UNDER_PAGE)) { markStale(); reopenToast(text); } else OE.toast(text, true);
        }
      }
      if (cleared) {
        if (onSaved) onSaved();
        try {
          const fresh = await OE.call(app, 'olmoearth_app_sheet', { labels_csv_path: data.path, face: to });
          // WORDING: r5b J21
          replaceEditor(fresh, 'Removed: the ' + answersOf(column) + '.');
        } catch (_) {
          // Nothing changed the file but this panel: the footer says the read
          // failed (W.notReadAgain), not that it changed elsewhere, beside
          // its Reopen button, which markStale hands the pressed button's
          // focus and which that line describes; the toast says only which
          // answers went, a plain message (r5-i B7: it said the failure a
          // second time, with the raw error, as an error).
          markStale('read');
          // WORDING: r5i B7
          reopenToast('The ' + answersOf(column) + ' were removed.', false);
        }
      }
    } finally {
      state.clearing = false;
      busyArea(false);
    }
    if (!stale) return;
    // Saved elsewhere since this page read it: take that in, then say so.
    state.syncing = true;
    let fresh = null;
    try { fresh = await readFresh(); } finally { state.syncing = false; }
    if (!fresh) return;
    adopt(fresh);
    if (!state.stale) {
      // WORDING: r5b J21
      OE.toast('Not cleared: the sheet was saved elsewhere since this panel read it, and it shows those labels now. '
        + 'Press the button again to remove the ' + answersOf(column) + '.', true);
    }
  }

  /* The two-map sheet's blind switch, at the reviewer's press: unticked, it
     asks the server for the two maps' classes once, and shows them; ticked,
     it hides them. Locked (r5-k K2: pairLocked), it stays ticked and asks
     nothing. */
  async function setBlind(on) {
    if (!on && pairLocked()) { blind.checked = true; state.blind = true; return; }
    if (on || state.revealed) { state.blind = on; render(); return; }
    if (state.revealing) return;
    state.revealing = true;
    blind.setAttribute('aria-busy', 'true');
    try {
      const fresh = await OE.call(app, 'olmoearth_app_sheet', { labels_csv_path: data.path, face, unblind: true });
      if (!fresh || fresh.sheet_id !== data.sheet_id || fresh.face !== face) {
        markStale();
        throw new Error(OE.CHANGED_UNDER_PAGE);
      }
      for (const w of Array.isArray(fresh.windows) ? fresh.windows : []) {
        if (w && byHandle.has(w.handle)) state.extra.set(w.handle, extraOf(w));
      }
      state.revealed = true;
    } catch (err) {
      const text = 'Not shown: ' + bare(err.message) + '.';
      // The sheet changed under the page (markStale above): a toast oeLeave may clear.
      if (state.stale && err.message === OE.CHANGED_UNDER_PAGE) reopenToast(text); else OE.toast(text, true);
    } finally {
      state.revealing = false;
      blind.removeAttribute('aria-busy');
      if (!state.revealed) blind.checked = true;
      state.blind = blind.checked;
      render();
    }
  }

  /* After a save whose answer says the server worked out more windows than
     were sent (rederived: on a seen sheet, a class typed into the CSV gets
     its right or wrong when any label is saved): the sheet is read again
     and taken in, every change made here winning, so the counts shown and
     told and the version the page holds name the same file. */
  async function rereadMine() {
    if (state.stale || state.syncing || state.saving) return;
    state.syncing = true;
    clearTimeout(timer);
    try {
      const fresh = await readFresh();
      if (fresh) adopt(fresh, { mine: true });
    } finally {
      state.syncing = false;
    }
    updateProgress();
    tellClaude();
    if (pending() && !state.stale) schedule();
  }

  /* Send the changed answers; an answer changed again during the save stays
     pending and goes in the next one. auto: the 1.5 s save after a change,
     which says nothing when nothing is left to send. While the sheet is read
     again (syncing), the save waits for the next turn. A failure asks for a
     press of Save labels, or, once this editor is off the page, names the
     windows not saved (tryAgain). quiet: the save when Claude closes the
     card (oeFlush, r4-k K1): no toast, no onSaved, no focus move, and no
     read of the sheet after it (a stale or rederived answer is left as
     it is; the server's version check keeps the file whole). */
  async function save({ auto = false, quiet = false } = {}) {
    clearTimeout(timer);
    if (state.saving) return;
    if (state.stale) {
      if (!quiet) reopenToast('Not saved: this sheet changed since the panel opened it. Reopen the sheet to go on.');
      return;
    }
    if (state.syncing) { schedule(); return; }
    if (!pending()) {
      // r5-k K5: a press of Save labels with nothing waiting is said under Save labels (footerSay).
      if (!auto && !quiet) footerSay(saveSaid, 'Every change is saved to ' + data.name + '.');
      return;
    }
    let failed = false;
    let stale = false;
    let rederived = false;
    let savedWords = '';  // r5-k K5: "Saved to ...", said beside the counter once updateProgress has shown its place
    const sent = new Map([...state.dirty].map((h) => [h, state.labels.get(h) || '']));
    firstUnsaved = 0;
    state.saving = true;
    quietSave = quiet;
    updateProgress();
    try {
      // The sheet's id, the face and the rule this page drew: the server
      // refuses the save if the sheet changed since; and the version this
      // page holds, refused as stale if the sheet was saved elsewhere since.
      const args = {
        labels_csv_path: data.path, sheet_id: data.sheet_id, face, rule,
        answers: [...sent].map(([h, value]) => ({ handle: h, answer: value })),
      };
      if (typeof data.version === 'string') args.version = data.version;
      const answer = await OE.call(app, 'olmoearth_app_save_labels', args);
      if (answer && typeof answer.version === 'string') data.version = answer.version;
      rederived = Boolean(answer && answer.rederived === true);
      for (const [h, value] of sent) {
        state.base.set(h, value);
        if (state.labels.get(h) === value) state.dirty.delete(h);
      }
      if (answer && 'mix_problem' in answer) {
        state.mix = answer.mix_problem
          ? { problem: String(answer.mix_problem), column: String(answer.clear_column || '') } : null;
      }
      staleResends = 0;
      state.saveFailed = false;
      if (!quiet) {
        // r5-k K5: beside the counter, in the place of "Saved automatically"
        // (counterSay, in finally: words written while that place is hidden,
        // as it is during the save, may go unsaid); the counter's line about
        // a save made elsewhere goes (its note on the squares kept rides on
        // these words).
        savedWords = 'Saved to ' + data.name + '.' + (keptNote ? ' ' + keptNote : '');
        countSaid.textContent = '';
        keptNote = '';
        if (onSaved) onSaved();
      }
    } catch (err) {
      failed = true;
      const said = bare(err.message || 'The save failed');
      // Save labels stays in view until a save goes through (B/sheet-finish-button).
      if (!err.stale) state.saveFailed = true;
      if (err.stale) {
        // Saved elsewhere since this page read it: read it, take it in, send again.
        stale = true;
      } else if (said.startsWith(OE.CHANGED_UNDER_PAGE)) {
        // A retry is refused the same way: the sheet must be read again.
        state.stale = true;
        if (!quiet) {
          OE.toast('Not saved: this sheet changed since the panel opened it. '
            + (wrap.isConnected ? 'Reopen the sheet to go on' : 'Open ' + data.name + ' again to go on')
            + '; the changes not saved here are not kept.', true);
        }
      } else if (!quiet) {
        footerSay(saveSaid, 'Not saved' + (wrap.isConnected ? '' : ' to ' + data.name) + ': ' + said + '. ' + tryAgain(), true);
      }
    } finally {
      state.saving = false;
      updateProgress();
      if (savedWords) counterSay(savedWords);
      quietSave = false;
      // Answers changed during the save go in the next one; a failed save waits
      // for the next change or a press of Save rather than retrying on its own.
      if (pending() && !failed) schedule();
      tellClaude();
    }
    if (quiet) return;
    if (stale) await resync();
    else if (rederived) await rereadMine();
  }

  /* The finish button, named for the sheet's job (estimate, certify, which
     map), in the footer from the first open (reader test 2's
     B/sheet-finish-button): greyed (aria-disabled, so it keeps the focus)
     until every window is labelled and saved (ready), then it sends the
     message that names the tool to call; the footer then says it was sent,
     or shows it to type when the host did not take it (OE.sendJob). Once
     ready, the line beside it says answers must not change after the press
     (W.everyDone, A/F17); on a sheet that already gave its result it says
     only that (W.everyDoneGraded, r5-d R06), as the notice over the sheet
     does, never that an answer may change until the press. While a window
     has no answer the same place says why the button is grey (W.notYet, r5-f
     F02; r5-e H1 took the lead's sentence on it away, and a grey button and
     a counter say it in no words): not while the sheet is stale (its own
     footer words), not while only a save is pending; on a sheet that already
     gave its result, whose answers are locked (r5-k K3), it says instead
     that a square has no answer and a new result needs a fresh sheet
     (W.gradedBlank: a square blanked elsewhere after the grading), never to
     answer one (r5-b J06: nothing under its notice asks for one). A
     server without the finish fields gets the old words. The button is made
     once and kept in place, so it keeps the focus when the progress is drawn
     again; once ready after the last answer it takes the focus, if the user
     has not moved it since. A held Enter sends the message once. A press
     while greyed does nothing; a press when ready first waits for any save,
     read or check under way, then asks the server whether the sheet was
     saved elsewhere (check, busy with aria-disabled meanwhile): one that
     was, or a take-in during the press that showed a change (changes), is
     taken in and nothing is sent, so the user sees what changed and presses
     again. One message per press (r5-n JS-1, sheet finish): the button stays
     busy (finishing, aria-disabled) until the host answers the message, so a
     press meanwhile does nothing (it was on again while the message went,
     and a second press asked Claude to grade twice), and the second click of
     a fast double-click (its detail 2) does nothing, even after the host
     answered the first; a key's press (detail 0) counts. */
  let finishButton = null;
  let finishing = false;
  // The line beside the finish button once every answer is saved (A/F17);
  // a sheet already graded says that it gave its result (r5-d R06).
  // WORDING: r5d R06
  const everyDoneLine = OE.el('span', { class: 'small' },
    data.graded && typeof data.graded === 'object' ? W.everyDoneGraded : W.everyDone);
  // In its place while a window has no answer yet (r5-f F02); on a sheet
  // already graded, whose answers are locked (r5-k K3), what is missing and
  // what a new result needs (W.gradedBlank), never to answer it. The finish
  // button names it as its description meanwhile (updateFinish), so a
  // screen reader that reaches the grey button hears why. r5-r N2: the line
  // ends with a text button that shows the next square with no answer
  // (goOpenLink, jumpOpen: what "Next unlabelled" does), then its full stop:
  // "Answer every square first: go to the next unlabelled." Not on a graded
  // sheet, which takes no answer.
  // WORDING: r5r
  const goOpenLink = OE.el('button', {
    class: 'link', style: LINE_LINK, 'data-focus': 'go-open', onclick: () => jumpOpen(),
  }, W.goOpen);
  // WORDING: r5f F02
  // WORDING: r5k K3 (gradedBlank on a graded sheet)
  const notYetLine = OE.el('span', { class: 'small', id: 'oe-not-yet-' + Math.random().toString(36).slice(2) },
    gaveResult ? W.gradedBlank : [W.notYet, goOpenLink, '.']);
  /* Whether the sheet can be sent for grading now: complete and saved. */
  const ready = () => complete() && !pending() && !state.stale;
  async function finishPress(message) {
    if (finishing || !ready()) return;
    finishing = true;
    finishButton.setAttribute('aria-disabled', 'true');
    let changed = false;
    try {
      const before = changes;
      while (state.saving || state.syncing) await wait(100);
      if (checking) await checking;
      changed = (await check({ force: true })) || changes !== before;
    } finally {
      finishing = false;
      finishButton.setAttribute('aria-disabled', String(!ready()));
    }
    if (changed || state.stale || pending() || !complete()) return;
    // r5-n JS-1 (sheet finish): busy again, with no wait since the finally
    // above, until the host answers the message below.
    finishing = true;
    finishButton.setAttribute('aria-disabled', 'true');
    // r5-k K2: the result was asked for once Claude took the message; a
    // two-map sheet's switch may then show what the two maps say. A message
    // the host did not take is shown to copy (resultShown): the switch stays
    // locked, its line saying when it opens then (W.pairLockedNotSent).
    // updateProgress also fits the scroll margin to the footer, which now
    // holds the press's line (r5-k K5, fitFoot).
    const sent = await OE.sendJob(app, message, finishType);
    if (sent) resultSent = true; else resultShown = true;
    updateProgress();
    finishing = false;
    finishButton.setAttribute('aria-disabled', String(!ready()));
  }
  /* The finish button's words (r5-m B5), ok: whether the sheet is ready
     now. The job's label (the server's finish_label); on a sheet that
     already gave its result, while it is ready, what its press shows: the
     result again (W.gradedFinish), or, its answers changed since that
     result (the page copy's graded changed_since_first), the withdrawn card
     (W.gradedChanged). A graded sheet that is not ready (a blank square,
     two sets of answers: nothing in the panel can make it ready, r5-k K3)
     keeps the job's label, grey, beside the line that says what is missing. */
  function finishLabel(ok) {
    if (!gaveResult || !ok) return data.finish_label || 'Ask Claude to grade it';
    return data.graded.changed_since_first === true ? W.gradedChanged : W.gradedFinish;
  }
  /* The finish button, made once (the server's message, sheet.finish_message;
     an older page copy without one gets the panel's own words). */
  function finishControl() {
    if (!finishButton) {
      const design = data.design_path ? ' with design_path ' + data.design_path : '';
      // The server's message (sheet.finish_message); an older page copy
      // without one gets this, in the sheet's unit word (the owner's ruling 6).
      // WORDING: squares
      const message = data.finish_message || ('I have labelled every ' + W.one + ' of the sheet ' + data.name
        + '. Estimate the map error from it: labels_path ' + data.path + design + '.');
      // The job's label (the server's finish_label); updateFinish gives a
      // sheet that already gave its result its own words once it is ready
      // (r5-m B5, finishLabel).
      finishButton = OE.el('button', {
        class: 'primary',
        'aria-disabled': 'true',
        // r5-n JS-1 (sheet finish): the second click of a double-click does nothing.
        onclick: (e) => { if (e.detail > 1) return; finishPress(message); },
        onkeydown: (e) => { if (e.repeat && (e.key === 'Enter' || e.key === ' ')) e.preventDefault(); },
      }, data.finish_label || 'Ask Claude to grade it');
    }
    return finishButton;
  }
  /* The footer's finish button as the counts stand: greyed until ready,
     with the A/F17 line beside it once ready (the button node stays in
     place, so it keeps the focus); once ready after the last answer, it
     takes the focus if the user has not moved it since. While a window has
     no answer, the line in that place says to answer every one first (r5-f
     F02; on a sheet that already gave its result, that one has no answer
     and a new result needs a fresh sheet: r5-k K3, W.gradedBlank): counted
     as the counter counts (count() < n), so it is not said on
     a sheet with no window, nor on one whose every window is answered but
     which holds two sets of answers (its notice says what is in the way).
     While it is said, the button's aria-describedby names it; else the
     button has none, so it never names a line that is not there. */
  function updateFinish() {
    const button = finishControl();
    const ok = ready();
    // WORDING: r5m B5 (the button's words as the sheet stands, finishLabel; the node stays, so it keeps the focus)
    const label = finishLabel(ok);
    if (button.textContent !== label) button.textContent = label;
    if (!finishing) button.setAttribute('aria-disabled', String(!ok));
    if (!done.contains(button)) done.replaceChildren(button);
    if (ok && !done.contains(everyDoneLine)) done.prepend(everyDoneLine);
    else if (!ok && done.contains(everyDoneLine)) everyDoneLine.remove();
    const waits = !ok && !state.stale && count() < n;
    if (waits && !done.contains(notYetLine)) done.prepend(notYetLine);
    else if (!waits && done.contains(notYetLine)) notYetLine.remove();
    if (waits) button.setAttribute('aria-describedby', notYetLine.id);
    else button.removeAttribute('aria-describedby');
    if (ok && state.finishFrom !== null && !quietSave) {
      const active = document.activeElement;
      const stayed = !active || active === document.body
        || (active.dataset && active.dataset.focus === state.finishFrom);
      state.finishFrom = null;
      if (stayed) OE.focusIfHere(button);
    }
  }

  // The blind switch: a two-map sheet only (the owner's O1 of 14 Nov 2026: a
  // class map's sheet never shows the map's answer, so it has no switch). A
  // seen sheet has nothing to show, and right or wrong shows the map's class.
  const blind = twoMap ? OE.el('input', {
    type: 'checkbox', id: 'oe-blind', checked: true,
    onchange: () => setBlind(blind.checked),
  }) : null;
  /* r5-k K2 (the owner's O2): the switch stays locked until every square is
     answered and saved and the sheet's result was asked for: the finish
     press whose message Claude took (resultSent), or a sheet that already
     gave its result. Locked, it is ticked, aria-disabled with the greyed
     busy look (common.js), and described by the line under its words
     (W.pairLocked), which shows only then; a press or a key on it (or on
     its label) changes nothing and asks the server for nothing. Unlocked, it
     works as before. A real change of an answer after the press, made here
     (setLabel) or taken in from elsewhere (adopt), undoes the press: the
     switch is locked again, and hides what it showed (syncPairLock), until
     a new press Claude takes. A press whose message the host did not take
     shows it to copy (resultShown): the switch stays locked, its line then
     W.pairLockedNotSent (a sheet that gave its result opened again, or drawn
     again once Claude graded it, check, is unlocked). Blind labelling is
     otherwise as it was. */
  let resultSent = false;
  let resultShown = false;
  const pairLocked = () => !(ready() && (resultSent || gaveResult));
  if (blind) blind.addEventListener('click', (e) => { if (pairLocked()) e.preventDefault(); });
  // Under the switch's words (r5-k K2, after review: beside them it wrapped to
  // a loose line 16 px below at 780 and 380 px): a row of its own (flex-basis
  // 100%), 4 px under the label (the row's gap, switchRow), its words from
  // the label's words' left edge (31 px: the box's 16 px, its 4 and 3 px
  // margins and the label's 8 px gap). Outside the label, so it keeps the
  // body's colour and the label's text stays the switch's words.
  const pairLockLine = blind ? OE.el('span', {
    class: 'small', id: 'oe-pair-lock-' + Math.random().toString(36).slice(2), style: 'flex-basis:100%;padding-left:31px',
  }, W.pairLocked) : null;
  /* The switch as the counts stand (run by render and updateProgress): its
     lock, look and line (W.pairLockedNotSent once a press's message was
     shown to copy, every answer still in and saved); true when it was
     unticked and is ticked again here, so the caller draws the editor again
     with the maps' answers hidden. */
  function syncPairLock() {
    if (!blind) return false;
    const lock = pairLocked();
    blind.setAttribute('aria-disabled', String(lock));
    if (lock) blind.setAttribute('aria-describedby', pairLockLine.id); else blind.removeAttribute('aria-describedby');
    const said = lock && ready() && resultShown && !resultSent ? W.pairLockedNotSent : W.pairLocked;
    if (pairLockLine.textContent !== said) pairLockLine.textContent = said;
    pairLockLine.hidden = !lock;
    switchLabel.classList.toggle('oe-locked', lock);
    if (!lock || state.blind) return false;
    blind.checked = true;
    state.blind = true;
    return true;
  }

  // One window or the list, a joined toggle at Studio's full size.
  const viewToggle = OE.toggle([['face', W.oneView], ['list', 'List']], state.view,
    (v) => setView(v), { label: W.howToShow, small: false });
  // Studio's three-way toggle at its full size (7px padding), over the list's rows.
  const filter = OE.toggle(
    [['any', 'Any'], ['unlabelled', 'Unlabelled'], ['labelled', 'Labelled']], state.filter,
    (v) => { state.filter = v; render(); }, { label: 'Show rows', small: false });
  const filterBox = OE.el('div', {}, filter);
  filterBox.hidden = state.view !== 'list';

  // What the two-map sheet's blind switch hides: what the two maps say for a
  // square; never a stratum, which no page is sent (the last round's F11).
  // In plain words (r5-b J28; the owner ruled on 18 Nov 2026 that it is
  // locked until every answer is in and sent, r5-k K2). r5-m B2: its
  // "(keep it on while you answer)" asked nothing once the lock came, and
  // the answers lock once the maps' answers are shown, so it is gone.
  // WORDING: r5m B2
  // WORDING: r4i F11; r5b J28
  const blindText = 'Hide what the two maps say';
  // The row of switches, only where one exists: the two-map sheet's blind
  // switch (the owner's O1 took the class sheet's switches away), and under
  // its words, while it is locked, the line that says when it opens (r5-k
  // K2): rows 4 px apart, columns 16 px.
  const switchLabel = blind ? OE.el('label', { class: 'row', style: 'gap:8px', for: 'oe-blind' }, blind, blindText) : null;
  const switchRow = blind ? OE.el('div', { class: 'row', style: 'gap:4px 16px' }, switchLabel, pairLockLine) : null;
  // In the Details fold (the last round's S1): what never reaches this panel.
  // The blind line of a sheet labelled with the class seen (a seen sheet's,
  // and a class map's, which has no switch: the owner's O1), the
  // right-or-wrong face's line, or, beside the two-map sheet's blind switch,
  // the code line; then the Street map line where the sheet has places to
  // open (A/F21).
  const sentLine = seen || face === 'class' ? W.blindSeen : wrongFace ? W.wrongLine : blind ? W.codeLine : '';
  const foldLines = [sentLine, data.has_place ? W.openMap : '']
    .filter(Boolean).map((text) => OE.el('div', { class: 'small' }, text));
  // A named scores map's sheet already labelled right or wrong says why it stays so.
  const ruleNote = data.rule_note ? OE.el('div', { class: 'small' }, String(data.rule_note)) : null;
  // The part of the map this sheet is for, under its name (A/sheet-10:
  // "West part of this map (1 of 2)"), when the server names one.
  const partLine = data.part && typeof data.part === 'object' && typeof data.part.heading === 'string'
    && data.part.heading ? OE.el('div', { class: 'small' }, data.part.heading) : null;
  // A sheet already graded: its answers are locked (the owner's extra 2;
  // r5-k K3, which locks them), at body size (r5-b J06: it was the smallest
  // text over the largest invitation to answer).
  const gradedNote = data.graded && typeof data.graded === 'object'
    ? OE.el('div', { class: 'card summary', role: 'note' }, W.graded) : null;
  // r5-k K3: it says the answers are locked, and is their description (lockAnswers).
  if (gradedNote) gradedNote.id = 'oe-graded-' + Math.random().toString(36).slice(2);
  // Why, in the Details fold (r5-e H5: the notice's middle clause, a reason
  // the user does not need in order to stop).
  const gradedWhy = gradedNote ? OE.el('div', { class: 'small' }, W.gradedWhy) : null;
  // The lines on top (the lead, r5-b J13; two since r5-e H1), before the
  // face's instruction, the rule and the answers. What this is (where the
  // map's answer is hidden: not on the right-or-wrong face); on a sheet that
  // can show no pictures its plain form, which promises no satellite picture
  // (finding 10), and on a sheet with places to open, squares or points, what
  // to judge from (reader test 2's E/sheet-1; r5-b J11: a sheet of squares
  // with places was told it had none). Then the face's one Cannot tell line
  // (S1: W.cannotTellLine, or a certificate's own need; none on a two-map
  // sheet). No third line (r5-e H1): that there is no result until every
  // answer is given repeated the greyed finish button and the counter. A
  // graded sheet keeps the first line only (r5-b J06): its notice says not
  // to change an answer, so nothing here invites one.
  const whatBare = data.has_place ? W.whatPlaced : (twoMap ? W.whatBarePair : W.whatBare);
  const leadWords = {
    what: wrongFace ? null : picturesPossible ? W.what : whatBare,
    next: gradedNote ? null : cannotLineText || null,
    cannot: null,
  };
  // A two-map sheet hides what both maps say (r5-b J28).
  if (twoMap && leadWords.what) leadWords.what = leadWords.what.replace(W.hiddenOne, W.hiddenPair);
  const leadLines = typeof OE.lead === 'function' ? OE.lead(leadWords) : null;
  // While the reviewer has unticked the switch and the face shows what the
  // two maps say, the first line says they are shown (W.shownPair; r5-b
  // J28). r5-m B2: the answers lock then, and the hidden form no longer
  // comes back with the switch: ticked again the line says the answers were
  // shown and are locked (W.seenPair; render). Only the hidden clause is
  // swapped; whatBarePair has none, so its tail (W.barePairTail) is.
  const pairWhat = twoMap && leadLines ? leadLines.querySelector('p.what') : null;
  // r5-m B2: once the maps' answers were shown it says the answers are
  // locked (pairShown, pairSeen), and is their description (lockAnswers).
  if (pairWhat) pairWhat.id = 'oe-pair-what-' + Math.random().toString(36).slice(2);
  const pairHidden = pairWhat ? pairWhat.textContent : '';
  const pairSwap = (s) => (pairHidden.includes(W.hiddenPair)
    ? pairHidden.replace(W.hiddenPair, s) : pairHidden.replace(W.barePairTail, s));
  const pairShown = pairSwap(W.shownPair);
  const pairSeen = pairSwap(W.seenPair);
  // Two example pictures under the owner's rule (r4-k K9), each captioned,
  // and the line on where they come from in the Details fold; only with the
  // server's examples (the owner's rule, both pictures on disk), else none.
  const examples = OE.ruleExamples(data);
  const ruleExamples = examples ? OE.examplePictures(examples) : null;
  const examplesNote = examples ? OE.el('div', { class: 'small' }, OE.EXAMPLES_NOTE(examples.year)) : null;
  // The labelling rule from the first open (A/sheet-2) and, on a face that
  // takes Cannot tell and can show pictures, the cloud line under it (r4-m
  // M8, after review: the page copy has the same rule as the pictures, both
  // imagery.rule_for, so the pane draws its own rule lines only where the
  // page copy has none, and this is where a labeller reads the line); then
  // the example pictures. They sit in one box with a border (r5-e H2: no
  // small line hangs loose under the rule; before, the scope was a line of
  // the block). Who set the rule and when (data.rule_scope) is history, in
  // the Details fold (scopeNote; since r5-l L1 no line after it says which
  // maps the rule is for); that the cloud mask is a guide and the labeller
  // decides (A/sheet-3; r5-p, before: that no date is marked clear) is in the fold too, only on a sheet that can show pictures:
  // one that cannot shows no date to mark. The rule itself is at body size,
  // "Rule: " in bold (r5-b J14: the one text needed at every answer was the
  // smallest on the card), and the block stands right above the answers and
  // above the list (wrap.append).
  const cloudOnFace = takesQ && picturesPossible;
  const ruleBlock = pageRule ? OE.el('div', { class: 'card stack oe-rule' },
    OE.el('div', {}, OE.el('strong', {}, 'Rule: '), pageRule),
    cloudOnFace ? OE.el('div', { class: 'small' }, W.cloudLine) : null,
    ruleExamples) : null;
  const clearNote = pageRule ? (picturesPossible ? OE.el('div', { class: 'small faint' }, W.clear) : null) : null;
  // The rule's scope, in the Details fold after the class line with no
  // period (r5-e H2; before, a small faint line in the rule block).
  const scopeNote = typeof data.rule_scope === 'string' && data.rule_scope ? OE.el('div', { class: 'small' }, data.rule_scope) : null;
  // No rule on the page copy, on a sheet that can show pictures (r5-b J26):
  // one line in the rule's place, at body size, from the first open (before,
  // the pictures pane said it, small and red, only once pictures were on). It
  // is hidden again if the pictures bring a rule the page copy could not read
  // (renderPreview's fallback rule lines). Not on a graded sheet (r5-b J06:
  // nothing under its notice invites an answer); a two-map sheet's names the
  // two maps (W.noRulePair).
  const noRuleLine = !pageRule && picturesPossible && !gradedNote
    ? OE.el('div', { class: 'oe-no-rule' }, twoMap ? W.noRulePair : W.noRule) : null;
  // That the answers are a reading of satellite pictures: in the
  // Details fold, only on a sheet that can show pictures (r5-b J13; before,
  // on the lead's third line).
  const readingNote = picturesPossible ? OE.el('div', { class: 'small' }, W.reading) : null;
  // The class face's line where the sheet does not know the map's period
  // (r5-d R08): in the fold, before what Cannot tell does, no longer a fourth
  // line on the face (faceInstructionText); its words are classWhat's own.
  // WORDING: r5d R08 (where it stands)
  const classNoPeriod = face === 'class' && !period ? OE.el('div', { class: 'small' }, W.classWhat(null)) : null;
  // ONE closed fold under the answers (the last round's S1): the sheet's file
  // name (it may be long: it wraps anywhere), the class face's line with no
  // period known (r5-d R08), who set the rule and when (scopeNote, r5-e H2),
  // why a graded sheet's answers must not change (gradedWhy, r5-e H5), the
  // instruction's rest and what
  // Cannot tell does, the keys, what never reaches this panel, the Street map
  // line, the clear-date note, where the example pictures come from (r4-k
  // K9; the sheet keeps its one fold), that the answers are a reading of
  // pictures (r5-b J13) and, on a sheet that holds two sets of answers, the
  // server's own sentence on them (mixNote, r5-b J21).
  // WORDING: r4h S1
  const foldLabel = 'Details';
  const foldParts = [
    OE.el('div', { class: 'mono', style: 'overflow-wrap:anywhere' }, data.name),
    classNoPeriod, scopeNote, gradedWhy,
    instruction, hint, ...foldLines, clearNote, examplesNote, readingNote, mixNote,
  ].filter(Boolean);
  const details = typeof OE.fold === 'function' ? OE.fold(foldLabel, ...foldParts)
    : OE.el('details', {}, OE.el('summary', {}, foldLabel), ...foldParts);

  // Annotation Lab's frame: one white section card, a teal title, the controls,
  // the face or the rows, the Details fold, and a footer with the counter and
  // the finish button. Before the answers the face says its lines (the lead,
  // r5-b J13; at most two, r5-e H1) and, right above the answers and above
  // the list, the rule in its box (r5-b J14, r5-e H2), or that none was set
  // (r5-b J26).
  wrap.append(
    OE.el('div', { class: 'stack' },
      OE.el('div', { class: 'spread', style: 'align-items:flex-start' },
        OE.el('div', {}, OE.el('h1', {}, 'Label sheet'), partLine),
        headerAction),
      gradedNote,
      leadLines,
      faceInstruction,
      ruleNote,
      switchRow,
      OE.el('div', { class: 'row', style: 'gap:16px' }, viewToggle, filterBox),
      notice,
      ruleBlock,
      noRuleLine,
      area,
      details),
    // r5-k K5: the save's own line (saveSaid) under the row with Save labels.
    OE.el('div', { class: 'sheet-footer spread' }, progress, OE.el('div', { class: 'row' }, done, saveButton), saveSaid,
      finishType));
  // One listener for the keys of every row; a pointer on a list is a pick, not browsing.
  body.addEventListener('keydown', onKey);
  body.addEventListener('pointerdown', () => { lastSelectKey = null; });
  // The face's keys, and a click inside it, which gives it the focus. A fold's summary keeps it (r5-k K7: Enter on
  // "What is sent, in full" opened the fold and sent the focus to the face; the face's keys still reach it there).
  faceEl.addEventListener('keydown', onFaceKey);
  faceEl.addEventListener('click', (e) => {
    const t = e.target;
    if (t && t.closest && !t.closest('button, select, input, textarea, a, label, summary')) OE.focusIfHere(faceEl);
  });
  // The page-wide listeners, removed together when the editor is replaced (oeDispose).
  const life = new AbortController();
  // r5-k K5: the footer's height follows the panel's width, the fonts and the host's full screen too: the scroll
  // margin follows it (fitFoot), until the editor is replaced.
  if (typeof ResizeObserver === 'function') {
    const footWatch = new ResizeObserver(() => fitFoot());
    footWatch.observe(wrap.querySelector(':scope > .sheet-footer'));
    life.signal.addEventListener('abort', () => footWatch.disconnect());
  }
  /* r5-k K5, after the check: a control the focus lands on ends above the
     footer. The browser scrolls a focused control into view only when it is
     out of view, so one already in view under the pinned footer stays hidden
     whatever its scroll margin: the next frame, it is measured and the page
     scrolled by what the footer covers and 8 px, never past the control's top
     (the footer's own controls are left alone). */
  wrap.addEventListener('focusin', (event) => {
    const target = event.target;
    const foot = wrap.querySelector(':scope > .sheet-footer');
    if (!foot || !(target instanceof Element) || foot.contains(target)) return;
    const next = typeof requestAnimationFrame === 'function' ? requestAnimationFrame : (fn) => setTimeout(fn, 16);
    next(() => {
      if (document.activeElement !== target || !foot.getClientRects().length) return;
      const at = target.getBoundingClientRect();
      const over = at.bottom + 8 - foot.getBoundingClientRect().top;
      if (over > 0 && at.top > 0) window.scrollBy(0, Math.min(over, at.top));
    });
  }, { signal: life.signal });
  // Shown again, or the panel gets the focus: was the sheet saved elsewhere?
  // (check: at most once every 3 s; an editor no longer on the page asks nothing.)
  const onShown = () => {
    if (wrap.isConnected && document.visibilityState === 'visible') check();
  };
  document.addEventListener('visibilitychange', onShown, { signal: life.signal });
  window.addEventListener('focus', onShown, { signal: life.signal });
  // Hidden or unloaded (full screen left, the chat closed or switched): the
  // answers waiting for the 1.5 s save are sent at once. Best effort: the save
  // is a host call, and a frame torn down may not get to send it.
  const flush = () => {
    if (wrap.isConnected && pending() && !state.saving && !state.syncing && !state.stale) save({ auto: true });
  };
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') flush();
  }, { signal: life.signal });
  window.addEventListener('pagehide', flush, { signal: life.signal });
  /* The host's display mode (OE.sheetMode): the face's dates are a grid in
     full screen, a strip inline. */
  wrap.oeMode = (mode) => { faceEl.classList.toggle('full', mode === 'fullscreen'); };
  /* For the page that swaps this editor for another (the Check card's
     zones): after any save or read in flight, save what is waiting; true
     when nothing is left to save here (every change saved, or the sheet
     changed under the page, so nothing can be: then at once, with no save
     and so no "Reopen the sheet" toast for an editor about to go; also when
     the save tried here is the one that finds the sheet changed), false
     when a save failed and the editor must stay. */
  wrap.oeSettle = async () => {
    while (state.saving || state.syncing) await wait(100);
    if (state.stale) return true;
    const ok = await settle();
    return ok || state.stale;
  };
  /* For that page, when its swap stopped because oeSettle gave false
     (r5-m B3, after review: the Check card's NOT_SWAPPED, a toast until
     then, said again under the footer's own "Press Save labels to try
     again."): its words, said once, in the footer's line under Save labels,
     as an error; after the failed save's reason that line keeps, in place
     of tryAgain's words, else alone. It goes as that line does: a save
     taken empties it (counterSay). */
  wrap.oeSay = (text) => {
    const tail = tryAgain();
    const was = saveSaid.classList.contains('error') ? saveSaid.textContent : '';
    footerSay(saveSaid, was.endsWith(tail) ? was.slice(0, -tail.length) + text : text, true);
  };
  /* For the page's teardown (r4-k K1: OE.start's onteardown, which gives
     the editors at most 3 s in all): after any save or read in flight,
     waiting at most FLUSH_WAIT_MS, save what is waiting, quietly (no toast,
     no onSaved, no focus move, nothing read again). True when nothing is
     left to save here; false while a save or read is still in flight, when
     the save failed, or once the sheet changed under the page. The
     visibilitychange and pagehide flush above stays the fallback. */
  const FLUSH_WAIT_MS = 2500;
  wrap.oeFlush = async () => {
    const until = Date.now() + FLUSH_WAIT_MS;
    while ((state.saving || state.syncing) && Date.now() < until) await wait(100);
    if (state.stale || state.saving || state.syncing) return false;
    if (pending()) await save({ auto: true, quiet: true });
    return !pending() && !state.stale;
  };
  /* For that page, before the swap: how many windows hold a change this
     editor cannot save (the sheet changed under the page), which the swap
     drops; 0 while it can still save. A count, never a window. */
  wrap.oeUnsaved = () => (state.stale ? pending() : 0);
  /* For that page, once this editor is swapped away (r4-l L3): an editor
     that stopped saving (the sheet changed under the page) with no change
     left unsaved clears its last toast about that (reopenToast: readFresh's,
     adopt's, the map link's, the clear's, the blind switch's, and save's on
     a press of Save then), while that toast is still the one shown: the
     sheet it names is no longer on the page, and the old line would stay
     under the new sheet. A toast about an unsaved change (the page's own
     count from oeUnsaved, or save's "the changes not saved here are not
     kept") and any other error stay. */
  wrap.oeLeave = () => {
    if (!state.stale || pending()) return;
    const bar = document.getElementById('oe-toast');
    if (bar && state.reopenSaid && bar.textContent === state.reopenSaid) OE.toast('');
  };
  /* The editor is replaced: its pictures timer and its page-wide listeners
     go, and an answer for pictures still on its way is dropped (imagesRun).
     The 1.5 s save is kept: it saves what a swap left unsaved. */
  wrap.oeDispose = () => {
    clearTimeout(picturesTimer);
    life.abort();
    state.imagesRun += 1;
    state.images.clear();
  };
  const ctx = app && typeof app.getHostContext === 'function' ? app.getHostContext() : null;
  wrap.oeMode(OE.displayMode || (ctx && ctx.displayMode) || 'inline');
  // The face opens at the first window not yet labelled, else the first.
  state.at = (windows.find((w) => !isLabelledRow(w)) || windows[0] || { handle: null }).handle;
  if (state.view === 'face') area.append(faceEl);
  else area.append(work);
  render();
  // Once the page has put the editor in place (right after this returns), Claude hears how far it is labelled.
  setTimeout(tellClaude, 0);
  return wrap;
};
