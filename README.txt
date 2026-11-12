BhashaCore - Multilingual FAQ Chatbot
======================================
Languages supported: English, Hindi, Bengali, Punjabi, Assamese, and Nepali.

Please see the comprehensive README.md for full documentation, architecture
details, automated testing, macOS setup, and language review instructions.

QUICK START (macOS):
--------------------
1. Create virtual environment:
   python3 -m venv .venv
   source .venv/bin/activate

2. Install minimal dependencies:
   python3 -m pip install -r requirements.txt

3. Run the interactive chatbot:
   streamlit run app.py

4. Run automated tests (31 tests):
   python3 -m unittest discover -s tests -v

5. Run multilingual evaluation benchmark (60 questions):
   python3 evaluate_faq.py

IMPORTANT NOTICE:
-----------------
The scholarship data is a fictional test fixture. Non-English translations
are unverified drafts and must be reviewed by native fluent speakers before
being presented as verified public guidance.
