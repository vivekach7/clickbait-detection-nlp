# A/B Test: Does showing a clickbait score change what people click?

A small browser-based experiment. Participants are shown 30 headlines (15 clickbait, 15 not, drawn from the model's predictions on the test set) across 3 rounds and asked to pick the 5 they're most interested in reading per round.

- **Control group**: sees plain headlines, no scores.
- **Experimental group**: sees the same headlines with a colour-coded badge showing the model's clickbait likelihood (Unlikely / Likely / Highly Likely / Clickbait).

Group assignment alternates per session. The hypothesis: if people are shown a clickbait warning, do they select fewer clickbait articles?

## Running it locally

This is static HTML/CSS/JS — no build step or server required for local testing:

```bash
cd ab_test
python3 -m http.server 8000
# open http://localhost:8000/projects.html
```

`main.js` expects `/partials/header.html` and `/partials/footer.html` to exist for the site nav — these aren't included here since they're part of the wider personal site this was embedded in, not the experiment itself. The experiment will still run without them; only the header/footer will be missing.

## How data was actually collected

Worth being upfront about this rather than letting the UI imply something more robust than it is: participant responses are written to **`localStorage`** in the participant's own browser (see `finalizeAndSave()` in `main.js`), not to a backend database. There's no server-side collection.

In practice, this meant manually exporting `localStorage.getItem('acbd_records')` from each participant's session after they completed the trial, then compiling those records into the dataset used in the results analysis. This works fine for a small, supervised study (19 participants) but doesn't scale and isn't suitable for unattended or remote data collection — there's no protection against a participant clearing their browser data before export, for instance.

If extending this into a larger study, the obvious next step would be replacing `localStorage.setItem` with a POST to a lightweight backend (even a simple serverless function writing to a database) so responses are captured immediately and centrally.

## Files

| File | Purpose |
|---|---|
| `projects.html` | The actual experiment page — headline list, round logic, badges |
| `main.js` | All experiment logic: group assignment, round splitting, selection limits, recording results |
| `master.css` | Styling, including the risk-band badge colours |
| `index.html`, `education.html`, `contact.html` | Other pages from the wider personal site this was hosted on |

## Known limitations

- Sample size (19 participants) is too small for statistical significance — results are directional only.
- `localStorage`-based collection (above).
- Group assignment alternates by session rather than using a proper randomisation scheme, so it isn't a true randomised controlled trial.
