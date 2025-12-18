# LDP Coach Matcher

**Live app:** [ldpcoachmatch-v2.streamlit.app](https://ldpcoachmatch-v2.streamlit.app/)

Match Vanderbilt MBA students with LDP executive coaches using AI-powered analysis.

---

## ✨ What It Does

This tool helps the LDP team create fair, thoughtful student-coach pairings:

- **Analyzes student profiles** - Reads resumes and survey responses to understand career goals, preferred coaching style, and background.
- **Analyzes coach bios** - Understands each coach's expertise, coaching style, and areas of focus.
- **Creates smart matches** - Uses AI to find the best coach for each student based on multiple factors.
- **Explains every match** - Generates a plain-language reason for each pairing.

---

## 🚀 How to Use

### Step 1: Log In
Enter your LDP team credentials to access the system.

### Step 2: Enter Your OpenAI API Key
You'll need to provide an OpenAI API key. [Get one here](https://platform.openai.com/api-keys) if you don't have one.  
*(Typical cost: less than $1 per matching run)*

### Step 3: Upload Files
- **Survey Excel** – Student survey responses (must have `Entry Id` column)
- **Student PDFs** – Resumes (filename should be the Entry ID, e.g., `123.pdf`)
- **Coach DOCXs** – Coach biography documents

### Step 4: Run Matching
Click "Run Matching" and watch the AI:
1. Analyze each coach's expertise and style
2. Analyze each student's goals and preferences
3. Calculate the best matches
4. Generate explanations for each pairing

### Step 5: Review & Export
- Edit matches in the table if needed
- Click "Recompute reasons" after edits
- Download the final Excel file

---

## 🎯 What Makes a Good Match?

The system considers:

| Factor | What It Means |
|--------|---------------|
| **Coaching Style** | Does the coach's approach match what the student wants? |
| **Industry Expertise** | Does the coach have relevant career experience? |
| **Background Alignment** | Do their profiles have common ground? |
| **Experience Level** | Is the coach right for the student's career stage? |
| **Special Fit** | International experience, DEI focus, etc. |

---

## 📋 Output

You'll get an Excel file with:

| Column | Description |
|--------|-------------|
| Entry Id | Student identifier |
| PrimaryCoach | Main coach assignment |
| PrimaryReason | Why this coach fits |
| SecondaryCoach | Backup coach (optional) |
| SecondaryReason | Why the backup fits |

---

## 🔒 Privacy & Security

- Login required to access
- Your API key is never stored permanently
- Student/coach data stays in your browser session
- Only essential text snippets are sent to OpenAI for analysis

---

## ❓ Questions?

Contact your LDP program administrator.
