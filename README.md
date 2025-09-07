LDP Coach Matcher

Live app: ldpcoachmatch-v2.streamlit.app

Match Vanderbilt business students to LDP coaches using resumes, survey responses, and coach bios.
The app uses embeddings + optimization to balance coach capacity, and (optionally) LLMs to generate natural language explanations.

🚀 How it works

Upload inputs in the sidebar:

Survey Excel (Sheet1 with a column Entry Id).

Student resumes (PDFs named <EntryId>.pdf).

Coach bios (DOCX, filename = coach alias).

(Optional) capacity CSVs (coach_alias,min,max).

Run Matching:

Text + theme similarity is computed with embeddings.

Each student gets a primary coach (min/max capacity).

Optionally, a secondary coach is also assigned.

Explanations:

With LLM toggled ON → one-sentence, natural reasons.

With LLM OFF → auto-generated template reasons.

Theme labels also adapt dynamically.

Review & Edit:

Interactive results table.

“Recompute reasons” after manual edits.

Download Excel with 5 columns:
Entry Id, PrimaryCoach, PrimaryReason, SecondaryCoach, SecondaryReason.
