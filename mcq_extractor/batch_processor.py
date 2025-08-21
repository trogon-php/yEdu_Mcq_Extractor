import pdfplumber
from .extractor import MCQExtractor
import hashlib
import re

class MCQBatchProcessor:
    def __init__(self, api_key):
        self.extractor = MCQExtractor(api_key)
        self.pages_per_batch = 10
        self.max_questions_to_ignore = 10
        self.overlap_pages = 2  # Pages to overlap when splitting batches

    def get_total_pages(self, pdf_path):
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    
    def process_pdf_in_batches(self, pdf_path, custom_prompt):
        print(f"\nProcessing PDF: {pdf_path}")

        if custom_prompt:
            print("Using personalized instructions.")
        else:
            print("Using default instructions.")

        pdf_file = self.extractor.upload_pdf(pdf_path)
        total_pages = self.get_total_pages(pdf_path)
        all_extracted_questions = []
        questions_to_ignore = []
        failed_batches = []  # Track failed batches for retry

        # First pass: Process all batches
        for i in range(0, total_pages, self.pages_per_batch):
            start_page = i + 1
            end_page = min(i + self.pages_per_batch, total_pages)
            print(f"\nProcessing pages {start_page}-{end_page}...")

            if questions_to_ignore:
                print(f"Ignoring {len(questions_to_ignore)} questions from the previous batch.")

            batch_results = self.extractor.extract_mcqs_from_pages(
                pdf_file, start_page, end_page, custom_prompt, questions_to_ignore
            )

            if batch_results:
                all_extracted_questions.extend(batch_results)
                questions_to_ignore = batch_results[-self.max_questions_to_ignore:]
                print(f"Extracted {len(batch_results)} questions. Total: {len(all_extracted_questions)}")
            else:
                print("No questions extracted from this batch.")
                # Add failed batch to retry list
                failed_batches.append({
                    'start_page': start_page,
                    'end_page': end_page,
                    'questions_to_ignore': questions_to_ignore.copy()
                })
                questions_to_ignore = []
        
        # Second pass: Retry failed batches by splitting them
        if failed_batches:
            print(f"\n🔄 Retrying {len(failed_batches)} failed batch(es) with split strategy...")
            
            for batch_info in failed_batches:
                start_page = batch_info['start_page']
                end_page = batch_info['end_page']
                questions_to_ignore = batch_info['questions_to_ignore']
                
                # Try splitting the failed batch
                split_results = self._retry_with_split_batch(
                    pdf_file, start_page, end_page, custom_prompt, questions_to_ignore
                )
                
                if split_results:
                    # Deduplicate questions before adding
                    deduped_results = self._deduplicate_questions(split_results, all_extracted_questions)
                    all_extracted_questions.extend(deduped_results)
                    print(f"✅ Split retry successful! Extracted {len(split_results)} questions ({len(deduped_results)} after dedup) from pages {start_page}-{end_page}")
                else:
                    print(f"❌ Split retry failed for pages {start_page}-{end_page}")
        
        # Renumber questions sequentially
        for i, question in enumerate(all_extracted_questions):
            question['SI.No'] = i + 1

        print(f"\nTotal questions extracted: {len(all_extracted_questions)}")
        return all_extracted_questions
    
    def _retry_with_split_batch(self, pdf_file, start_page, end_page, custom_prompt, questions_to_ignore):
        """Retry a failed batch by splitting it into smaller parts with overlap"""
        print(f"\nSplitting failed batch {start_page}-{end_page} into smaller parts...")
        
        # Calculate split points
        total_pages = end_page - start_page + 1
        if total_pages <= 3:  # Too small to split meaningfully
            print(f"Batch too small to split ({total_pages} pages). Trying once more as-is.")
            return self.extractor.extract_mcqs_from_pages(
                pdf_file, start_page, end_page, custom_prompt, questions_to_ignore
            )
        
        mid_point = start_page + (total_pages // 2)
        
        # First half: start_page to mid_point + overlap
        first_end = min(mid_point + self.overlap_pages, end_page)
        print(f"  Part 1: pages {start_page}-{first_end}")
        
        first_half = self.extractor.extract_mcqs_from_pages(
            pdf_file, start_page, first_end, custom_prompt, questions_to_ignore
        )
        
        # Second half: mid_point - overlap to end_page
        second_start = max(mid_point - self.overlap_pages, start_page)
        print(f"  Part 2: pages {second_start}-{end_page}")
        
        # For second half, use questions from first half as ignore list if available
        ignore_for_second = questions_to_ignore.copy() if questions_to_ignore else []
        if first_half:
            ignore_for_second.extend(first_half[-self.max_questions_to_ignore:])
        
        second_half = self.extractor.extract_mcqs_from_pages(
            pdf_file, second_start, end_page, custom_prompt, ignore_for_second
        )
        
        # Combine results
        combined_results = []
        if first_half:
            combined_results.extend(first_half)
            print(f"    Part 1 extracted: {len(first_half)} questions")
        
        if second_half:
            combined_results.extend(second_half)
            print(f"    Part 2 extracted: {len(second_half)} questions")
        
        return combined_results
    
    def _deduplicate_questions(self, new_questions, existing_questions):
        """Remove duplicate questions based on question content similarity"""
        if not new_questions or not existing_questions:
            return new_questions
        
        def normalize_question_text(question_obj):
            """Extract and normalize question text for comparison"""
            if not isinstance(question_obj, dict) or 'question' not in question_obj:
                return ""
            
            # Extract text from HTML and normalize
            text = question_obj['question']
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', ' ', text)
            # Normalize whitespace and convert to lowercase
            text = re.sub(r'\s+', ' ', text.lower().strip())
            return text
        
        def questions_similar(q1, q2, threshold=0.8):
            """Check if two questions are similar enough to be considered duplicates"""
            text1 = normalize_question_text(q1)
            text2 = normalize_question_text(q2)
            
            if not text1 or not text2:
                return False
            
            # Simple similarity check: if 80% of words match
            words1 = set(text1.split())
            words2 = set(text2.split())
            
            if not words1 or not words2:
                return False
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            similarity = intersection / union if union > 0 else 0
            return similarity >= threshold
        
        # Find duplicates
        deduplicated = []
        for new_q in new_questions:
            is_duplicate = False
            for existing_q in existing_questions:
                if questions_similar(new_q, existing_q):
                    print(f"    Removing duplicate question: {normalize_question_text(new_q)[:50]}...")
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(new_q)
        
        if len(deduplicated) < len(new_questions):
            print(f"    Deduplication: {len(new_questions)} -> {len(deduplicated)} questions")
        
        return deduplicated
        