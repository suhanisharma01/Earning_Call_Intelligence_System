# Earning Call Intelligence System

An AI-powered system for analyzing earnings call transcripts to extract key insights, themes, risks, and guidance using OpenAI's GPT models.

## Features

- **Data Ingestion**: Process and store earnings call transcripts from Motley Fool data into a SQLite database.
- **AI Analysis**: Generate intelligent reports including summaries, key themes, risk flags, and future guidance using OpenAI GPT models.
- **FastAPI Backend**: RESTful API for managing and analyzing earnings calls.
- **Streamlit Frontend**: User-friendly web interface for browsing calls, viewing transcripts, and generating reports.
- **Caching**: Reports are cached in the database to avoid redundant API calls.

## Project Structure

```
.
├── API.py                 # FastAPI application with endpoints for analysis
├── ingestion.py           # Script to ingest data from pickle file into SQLite
├── requirements.txt       # Python dependencies
├── api/
│   ├── analysis.py        # Analysis functions using OpenAI
│   └── streamlit_app.py   # Streamlit web application
│   ├── LICENSE
│   └── README.md
└── earnings_calls.db      # SQLite database (generated)
```

## Installation

1. Clone or download the project.

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `ANALYSIS_PROVIDER`: Set to "openai"
   - `OPENAI_MODEL`: Model to use (default: gpt-4o-mini)

## Data Preparation

1. Obtain the Motley Fool earnings call data as a pickle file (`motley-fool-data.pkl`).

2. Run the ingestion script:
   ```bash
   python ingestion.py
   ```
   This will create `earnings_calls.db` with the processed data.

## Usage

### Running the API

Start the FastAPI server:
```bash
python API.py
```
The API will be available at `http://127.0.0.1:8000`.

### Running the Streamlit App

Start the Streamlit application:
```bash
streamlit run api/streamlit_app.py
```
Access the app at `http://localhost:8501`.

## API Endpoints

- `GET /calls`: List earnings calls with filtering options.
- `GET /calls/{call_id}`: Get details of a specific call.
- `POST /analyze/{call_id}`: Generate AI analysis for a call.
- `GET /insights/{call_id}`: Retrieve cached analysis.

## Contributing

Contributions are welcome. Please ensure code follows Python best practices and includes appropriate tests.

## License

See [LICENSE](Earning_Call_Intelligence_System/LICENSE) for details.</content>
<parameter name="filePath">c:\Users\suhan\OneDrive\Desktop\Python projects\earning call intelligence system\README.md