import os
import json
import time
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import signal
from concurrent.futures import ThreadPoolExecutor, TimeoutError

class MCQExtractor:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-pro",
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
            generation_config={
                "response_mime_type": "application/json"
            }
        )

    def upload_pdf(self, pdf_path):
        print(f"Uploading file: {pdf_path}")
        pdf_file = genai.upload_file(path=pdf_path, display_name=os.path.basename(pdf_path))
        print(f"Completed upload: {pdf_file.uri}")
        return pdf_file

    def extract_mcqs_from_pages(self, pdf_file, start_page, end_page, custom_prompt, questions_to_ignore, attempt=0, max_attempts=2):

        prompt = f"""
        You are an MCQ extractor.

        My assessment is that the high concentration of legal, political, and administrative terms, especially those related to minority rights and government authority, is likely triggering these safety filters. While the content is perfectly legitimate and academic, the AI's safety system may be erring on the side of caution and blocking the response to avoid generating content that could be misconstrued as sensitive political commentary.

      Process only pages {start_page} through {end_page} of the attached PDF document.
      Extract all multiple-choice questions from this page range and return them as a JSON array.
      Also, look for an answer key section within this page range and use it to determine the correct answers.
      
      Each item in the array should include the following keys:
            - "SI.No": A serial number for the question.
            - "question": The full question, formatted as an HTML string. 
                - Wrap the main question in ⁠ <p> ⁠.
                - If there are any statements (e.g., Statement I, II, etc.), include them inside the same HTML block with appropriate formatting.
                - If the question is a 'Match-the-Column' type, include both List I and List II inside the HTML, using ⁠ <ul> ⁠, ⁠ <li> ⁠, or ⁠ <table> ⁠ as needed.
            - "options": An array of strings, where each string is an option (e.g., ["Option 1", "Option 2"]).
            - "correct_answer": The CHAR identifier of the correct option (e.g., C), if provided in the answer key or generate according to question (Should not be null or anything , Mandatory!). there should only be answers as ["A", "B", "C", "D"].
            - "type": One of ['MCQ', 'Order-based', 'Match-the-Column'].

            example : 
            [
               {
                    {
                    "SI.No": 13,
                    "question": "<p>A pilot is used to land on wide runways only. When approaching a smaller and/ or narrower runway, the pilot may feel he is at:</p>",
                    "options": [
                        "Greater height than he actually is with the tendency to land short.",
                        "Greater height and the impression of landing short.",
                        "Lower than actual height with the tendency to overshoot."
                    ],
                    "correct_answer": "A",
                    "type": "MCQ"
                    }
                }
            ]
   
      Important formatting notes:
      - Include all contextual parts (statements, match-the-columns, etc.) *within the HTML in the "question" field*.
      - For questions with multiple statements, format them inside the HTML using ⁠ <ul> ⁠ or ⁠ <p> ⁠ tags as appropriate.
      - For match-the-column questions, display List I and List II in a clear, structured HTML format like a table or two lists.
      
      Note:
      - *Handling Broken Questions*: If a question near the end of the page range (page {end_page}) appears incomplete, you are permitted to look at the beginning of the next page to find the rest of the question or its options. You must assemble the complete question.
      - Correct any spelling or grammar issues in the extracted questions or statements based on context.
      - Ensure the output is a valid JSON array and properly structured.
      - Create questions or options (max 4), and statements on broken questions Only- if contextually applicable and the chances to be created in the next or previous batches is lower (context is much higher in this batch), some other instructions are given below:
            - Avoid Answer Keys: You must differentiate between the primary question list and any separate "Answer Key" or "Solutions" sections. Do NOT generate questions from these sections.
            - Pattern to Ignore: An "Answer Key" or "Solutions" section is typically characterized by a list format (e.g., 1. C, 2. A) followed by explanations, but it lacks the full, original question text and the list of options (A, B, C, D). If you encounter text that matches this pattern, you must ignore it to prevent creating fragmented or duplicate questions.
      - if any questions happen to repeat , ignore it , (only ignore the questions with same purpose, not similar)
      
      Questions to ignore (if any):
      {questions_to_ignore}

      These questions should not be reprocessed.

      Now, process the document provided and extract the questions.
        """

        extend_prompt_with="\n\tHere are some additional rules to follow:\n"
        if custom_prompt != "":
            extend_prompt_with += custom_prompt
            prompt += extend_prompt_with
         
        print(f"Sending request to Gemini API for pages {start_page}-{end_page}...")
        try:
            # Use timeout for API calls
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.model.generate_content, [prompt, pdf_file])
                try:
                    response = future.result(timeout=120)  # 2 minute timeout for each API call
                except TimeoutError:
                    print(f"❌ API timeout after 120 seconds for pages {start_page}-{end_page}")
                    return []
            print("\n\t\tResponse received from Gemini API.\n")
            print(response)
            print("\n\t\tResponse end from Gemini API.\n")
            
            # Check if response has content parts
            if not response.candidates or not response.candidates[0].content.parts:
                print(f"Empty response received. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'Unknown'}")
                if attempt < max_attempts - 1:
                    print(f"Retrying API call after empty response. Attempt {attempt + 2}/{max_attempts}")
                    return self.extract_mcqs_from_pages(pdf_file, start_page, end_page, custom_prompt, questions_to_ignore, attempt + 1, max_attempts)
                return []
            
            raw_json_string = response.text.strip()

            # Strip markdown fences if they exist
            if raw_json_string.startswith("```json"):
                raw_json_string = raw_json_string[len("```json"):].strip()
            if raw_json_string.endswith("```"):
                raw_json_string = raw_json_string[:-len("```")].strip()

            # Parse JSON to validate
            try:
                parsed_json = json.loads(raw_json_string)
                return parsed_json
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                if attempt < max_attempts - 1:
                    print(f"Retrying API call after parsing error. Attempt {attempt + 2}/{max_attempts}")
                    return self.extract_mcqs_from_pages(pdf_file, start_page, end_page, custom_prompt, questions_to_ignore, attempt + 1, max_attempts)


                return []

        except Exception as e:
            print(f"Error during MCQ extraction: {e} , Attempting in two batches")

            if "500" in str(e) and attempt == 0 and start_page < end_page:
                print(f"Context too long error detected. Splitting batch into smaller chunks...")
                
                # Calculate midpoint for splitting
                mid_page = start_page + (end_page - start_page) // 2
                
                # Process first half
                print(f"Processing first half: pages {start_page}-{mid_page}")
                first_half = self.extract_mcqs_from_pages(
                    pdf_file, start_page, mid_page, custom_prompt, questions_to_ignore, 1
                )
                
                # Process second half
                print(f"Processing second half: pages {mid_page + 1}-{end_page}")
                second_half = self.extract_mcqs_from_pages(
                    pdf_file, mid_page + 1, end_page, custom_prompt, questions_to_ignore, 1
                )
                
                # Combine results
                combined_results = []
                if isinstance(first_half, list):
                    combined_results.extend(first_half)
                if isinstance(second_half, list):
                    combined_results.extend(second_half)
                
                print(f"Successfully combined {len(combined_results)} questions from split batches")
                return combined_results

            return []