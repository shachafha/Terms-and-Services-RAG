
# RAG Terms and Services Project

This project processes Terms and Services data using a Retrieval-Augmented Generation (RAG) model to answer questions about the data. The application runs on an Azure machine with Streamlit for the front-end interface.

## Getting Started

### Prerequisites
- An Azure virtual machine
- SSH access to the machine
- `conda` for managing Python environments

### Steps to Run the Project

1. **Turn on the Azure machine**  
   Make sure your Azure machine is powered on and accessible.

2. **Connect to the machine using SSH with port forwarding**  
   Open your terminal and run the following command, replacing `MACHINE_DNS` with your machine's DNS:

   ```bash
   ssh -L 8501:localhost:8501 student@MACHINE_DNS
   ```

3. **Clone the project repository**  
   If the project is not already on the machine, clone the repository using the following command:

   ```bash
   git clone https://github.com/your-repository-url.git
   ```

   Skip this step if the project is already cloned.

4. **Navigate to the project directory**  
   Change into the project directory:

   ```bash
   cd Terms-and-Services-RAG
   
   ```

5. **Activate the conda environment**  
   Run the following command to activate the existing environment:

   ```bash
   conda activate RagTermsAndServices
   ```

   If the environment does not exist, create it using the `requirements.txt` file provided:

   ```bash
   conda create -n RagTermsAndServices python=3.9.18
   conda activate RagTermsAndServices
   pip install -r requirements.txt
   ```

6. **Run the Streamlit app**  
   Start the Streamlit app with the following command:

   ```bash
   cd src
   streamlit run app.py --server.port 8501 --server.enableCORS false
   ```

7. **Access the app**  
   Open a web browser and go to:

   ```text
   http://localhost:8501/
   ```

## Project Structure
- `app.py`: The main Streamlit application file.
- `requirements.txt`: Contains the necessary dependencies for setting up the environment.

## Troubleshooting
- **Environment issues**: If the conda environment does not activate, check that `azureml_py38_PT_and_TF` is properly installed, or recreate it using the `requirements.txt`.
- **Connection issues**: Ensure that you use the correct `MACHINE_DNS` and that SSH port forwarding is properly set up.

## Example Questions for the RAG Model
- Choose company:Zalando, ask: What is Zalando's policy on returning items purchased from their partner merchants? 
- Choose company:Bolt, ask: What is the policy on retaining driver data after their Bolt account is closed?
- Choose company:Instagram, ask: What is the maximum time it can take for Instagram to respond to a request to review a removed post? 

