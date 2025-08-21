from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form ,HTTPException
from fastapi.responses import JSONResponse, FileResponse ,RedirectResponse
from fastapi.staticfiles import StaticFiles
import os
from mcq_extractor.batch_processor import MCQBatchProcessor
import uuid
from datetime import datetime , timezone
import json 
from dotenv import load_dotenv
from cleanup import cleanup_files
import sys
import tempfile
from docx import Document
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError

app = FastAPI()
load_dotenv()
api_key = os.getenv("API_KEY")


########################################## CALLING CONVERSION FUNCTION ##########################################
def convert_docx_to_pdf_custom(docx_path, pdf_path):
    """Convert DOCX to PDF using python-docx and reportlab"""
    try:
        # Read the DOCX file
        doc = Document(docx_path)
        
        # Create PDF document
        pdf_doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Extract text from DOCX and add to PDF
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():  # Only add non-empty paragraphs
                p = Paragraph(paragraph.text, styles['Normal'])
                story.append(p)
                story.append(Spacer(1, 12))  # Add space between paragraphs
        
        # Build the PDF
        pdf_doc.build(story)
        return True
        
    except Exception as e:
        print(f"Error in custom DOCX to PDF conversion: {e}")
        return False

# function for the conversion process call 
def process_with_timeout(file_path: str, result_file_name: str, uuid: str, customInput: str, timeout_seconds=300):
    """Process file with timeout handling"""
    def target_function():
        return process_file_core(file_path, result_file_name, uuid, customInput)
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(target_function)
        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except TimeoutError:
            print(f"❌ Processing timeout after {timeout_seconds} seconds")
            update_metadata(uuid, "Error: Processing timeout")
            # Clean up files
            if os.path.exists(file_path):
                os.remove(file_path)
            pdf_path = file_path.replace('.docx', '.pdf') if file_path.endswith('.docx') else file_path
            if pdf_path != file_path and os.path.exists(pdf_path):
                os.remove(pdf_path)
            return None
        except Exception as e:
            print(f"❌ Unexpected error during processing: {e}")
            update_metadata(uuid, f"Error: {str(e)}")
            return None

def process_file_core(file_path: str, result_file_name: str, uuid: str, customInput: str):
    """Core processing logic without timeout wrapper"""
    if not os.path.isfile(file_path):
        print(f"❌ File not found: {file_path}")
        update_metadata(uuid, "Error: File not found")
        return
    
    # Check if the file is a DOCX and needs conversion
    pdf_path = file_path
    if file_path.endswith('.docx'):
        # Convert DOCX to PDF
        pdf_path = file_path.replace('.docx', '.pdf')
        try:
            success = convert_docx_to_pdf_custom(file_path, pdf_path)
            if success:
                print(f"✅ Converted DOCX to PDF: {pdf_path}")
                # Remove the original DOCX file
                os.remove(file_path)
            else:
                print(f"❌ Error converting DOCX to PDF")
                update_metadata(uuid, "Error: Failed to convert DOCX to PDF")
                if os.path.exists(file_path):
                    os.remove(file_path)
                return
        except Exception as e:
            print(f"❌ Exception during DOCX conversion: {e}")
            update_metadata(uuid, f"Error: DOCX conversion failed - {str(e)}")
            if os.path.exists(file_path):
                os.remove(file_path)
            return
    
    try:
        # Process the PDF for MCQ extraction
        processor = MCQBatchProcessor(api_key)
        questions = processor.process_pdf_in_batches(pdf_path, customInput)
       
        if questions == []:
            print("No questions found ")
            update_metadata(uuid, "Processed, No questions found")
        else:
            print(f"Successfully extracted {len(questions)} questions")
        
        # Write result file
        final_json = json.dumps(questions, indent=2, ensure_ascii=False)
        with open(result_file_name, "w", encoding='utf-8') as f:
            f.write(final_json)

        print(f"Processing done, result saved as {result_file_name}")
        update_metadata(uuid, "Processed")
        
    except Exception as e:
        print(f"❌ Error during MCQ processing: {e}")
        update_metadata(uuid, f"Error: MCQ processing failed - {str(e)}")
    finally:
        # Clean up the PDF file
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

def process_file(file_path: str, result_file_name: str, uuid: str, customInput: str):
    """Main process file function with timeout and error handling"""
    return process_with_timeout(file_path, result_file_name, uuid, customInput, timeout_seconds=300)

########################################## UPDATE METADATA ##########################################
def update_metadata(uuid: str, status: str):
    print("update")
    with open("metadata/metadata_list.json", "r") as f:
        metadata_list = json.load(f)
    
    # Find and update specific metadata
    for metadata in metadata_list:
         if metadata["uuid"] == uuid:
               metadata.update({"status": status})
               break
    
    # Write back to file
    with open("metadata/metadata_list.json", "w") as f:
         json.dump(metadata_list, f, indent=2)

########################################## SAVING METADATA ##########################################
def save_metadata(metadata):
    print("SAVING")
    try:
        with open("metadata/metadata_list.json", "r") as f:
            existing_data = json.load(f)
            if not isinstance(existing_data, list):
                existing_data = []
    except json.JSONDecodeError:
        existing_data = []

    existing_data.append(metadata)

    with open("metadata/metadata_list.json", "w") as f:
        json.dump(existing_data, f, indent=4)

########################################## LOADING METADATA ##########################################
@app.get("/metadata")
def load_metadata():
    try:
        with open("metadata/metadata_list.json", "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

########################################## Mount static files ##########################################
directory = os.path.join(os.path.dirname(__file__), 'static')
app.mount('/static', StaticFiles(directory=directory), name='static')

########################################## Ensure uploads directory exists ##########################################
uploadSave_Directory = os.path.join(os.path.dirname(__file__), 'tempUploads')
os.makedirs(uploadSave_Directory, exist_ok=True)

@app.get('/')
async def read_root():
    return JSONResponse(content={'message': 'Welcome to the MCQ Extractor!'}, status_code=200)

##########################################  ##########################################
@app.get("/upload")
async def serve_index():
    # Serve the 'index.html' file from the 'static' folder
    return FileResponse(os.path.join("static", "index.html"))

########################################## Upload file ##########################################
@app.post('/upload')
async def upload_file( background_tasks: BackgroundTasks, customInput: str = Form(...), file: UploadFile = File(...) ):
    # clean files older than 24 hours
    cleanup_files()

    # Check if uploaded file is PDF or DOCX
    if not (file.filename.lower().endswith('.pdf') or file.filename.lower().endswith('.docx')):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")

    unique_id = str(uuid.uuid4())
    
    # Keep original extension for processing
    original_extension = '.pdf' if file.filename.lower().endswith('.pdf') else '.docx'
    file_name = unique_id + original_extension
    file_location = os.path.join(uploadSave_Directory, file_name ) # file.filename is the name of the file , name is taken from the temporary storage of UploadFile
    with open(file_location, 'wb') as buffer:
        buffer.write(file.file.read())

    json_file_name = os.path.join(os.path.dirname(__file__)+"/Outputs/"+unique_id+".json")

    metadata = {
        "uuid": unique_id,
        "original_filename": file.filename,
        "pdf_filename": file_name,
        "json_filename": json_file_name,
        "status":"Processing",
        "upload_timestamp":  datetime.now().astimezone().isoformat()
    }
    save_metadata(metadata)

    print("Before adding background task")
    background_tasks.add_task(process_file, file_location, json_file_name, unique_id, customInput)
    print("After adding background task")

    return RedirectResponse(url=f"""metadata/{unique_id}""",status_code=302)

@app.get("/metadata/{uuid}")
async def get_status(uuid: str):
    print("calling metadata function for uuid: ", uuid)
    metadata_list = load_metadata()
    metadata = next((item for item in metadata_list if item["uuid"] == uuid), None)
    if not metadata:
        return JSONResponse(content={"status":0,"message":"Metadata not found","metadata":[]},status_code=200)

    if metadata["status"] == "Processing":
        return JSONResponse(content={"status":1,"message":"Processing in progress","metadata":metadata},status_code=200)
    
    # Check if there was an error
    if metadata["status"].startswith("Error:"):
        return JSONResponse(content={"status":2,"message":metadata["status"],"metadata":metadata},status_code=200)
    
    return JSONResponse(content={"status":1,'message': 'File Proccessed Succesfully', 'metadata': metadata}, status_code=200)

@app.get("/json/{uuid}")
async def get_json(uuid:str):
    print("getting json data for uuid :" ,uuid)
    metadata_list=load_metadata()
    metadata = next((item for item in metadata_list if item["uuid"] == uuid), None)

    if not metadata:
        return JSONResponse(content={"status":0,"message":"Metadata not found","metadata":[]},status_code=200)

    # Check if the file exists
    if metadata["status"] == "Processing":
        return JSONResponse(content={"status":1,"message":"Processing in progress","data":[]},status_code=200)
    
    # Check if there was an error during processing
    if metadata["status"].startswith("Error:"):
        return JSONResponse(content={"status":2,"message":metadata["status"],"data":[]},status_code=200)
    
    if not os.path.isfile(metadata["json_filename"]):
        print(f"❌ File not found: {metadata["json_filename"]}")
        return JSONResponse(content={"status":2,"message":"File not found","data":[]},status_code=200)
    
    with open(metadata["json_filename"], "r", encoding="utf-8") as f:
        data = json.load(f)

    return JSONResponse(content={"status":1,'message': 'File Processed Succesfully',"data":data},status_code=200)
