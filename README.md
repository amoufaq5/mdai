# Multimodal Diagnostic AI

## Overview
This project implements an AI-based diagnostic system that processes multimodal data (text, images, numerical) to diagnose common diseases and illnesses. It integrates clinical mnemonics (ASMETHOD, WWHAM, ENCORE, SIT DOWN SIR) for data collection and decision-making. Based on the diagnosis, the system will recommend OTC medication or refer the patient to a doctor.

## Project Structure
- **main.py**: Contains the FastAPI REST API endpoints for patient interaction.
- **model.py**: Defines the TensorFlow multimodal model.
- **data_preprocessing.py**: Loads and preprocesses CSV/JSON files with a mapping system to standardize parameters.
- **evaluation.py**: Implements heuristic diagnostic rules using clinical mnemonics.
- **train.py**: Script to train the model on your local data.
- **utils.py**: Utility functions for data loading and normalization.
- **requirements.txt**: Lists required Python packages.

## Setup Instructions
1. **Clone the Repository & Setup Environment**
   ```bash
   git clone <your-repo-url>
   cd project
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   pip install -r requirements.txt
