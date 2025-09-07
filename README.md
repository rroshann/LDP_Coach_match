Always show details
import pypandoc

# Short, non-technical README text
readme_text = """# LDP Coach Matcher

**Live app:** [ldpcoachmatch-v2.streamlit.app](https://ldpcoachmatch-v2.streamlit.app/)  

This tool helps match Vanderbilt business students to LDP coaches using student resumes, survey responses, and coach bios.  
It creates fair, balanced assignments while also providing plain-language reasons for each match.

---

## 🚀 How it works
1. **Upload files** in the sidebar:
   - Survey Excel file (`Sheet1` with a column `Entry Id`).
   - Student resumes (PDFs named `<EntryId>.pdf`).
   - Coach bios (DOCX files, filename = coach alias).
   - *(Optional)* coach capacity CSVs (`coach_alias,min,max`).

2. **Run Matching**:
   - The system analyzes text to measure similarities between students and coaches.
   - Each student is matched with a **primary coach**.
   - Optionally, a **secondary coach** is also assigned.

3. **Explanations**:
   - If AI is enabled → you get natural, fluent reasons for each match.
   - If AI is off → you still get simple, clear template reasons.
   - Theme labels adapt dynamically from your data.

4. **Review & Edit**:
   - Results appear in an interactive table.
   - You can make manual changes, then click **Recompute reasons**.
   - Download the final Excel file with 5 columns:  
     `Entry Id, PrimaryCoach, PrimaryReason, SecondaryCoach, SecondaryReason`.

---

## 📄 Notes
- Matches are aligned by `Entry Id`, not file order.
- Documents remain local; only short keyphrases or theme names are sent to AI (if enabled).
- Typical runs (≈70 students, 40 coaches) are very low cost if AI features are used.
