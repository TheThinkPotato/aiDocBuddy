# AI Doc Buddy  Python Project Setup Guide

This guide explains how to set up and run the project.

## Prerequisites

Before starting, ensure you have the following installed:

- **Python 3**
- **pip3**
- **ollama**
  - **deepseek-r1 model installed**
  - **mxbai-embed-large model installed**

## Install Python Dependencies

Install all required Python libraries using `requirements.txt`:

```bash
pip3 install -r requirements.txt
```

## Usage

To run the project, use the following command:

```bash
python ./aiDoc.py ./doc.pdf
```

- Replace `./doc.pdf` with the path to your input file.
- The first parameter is the name of the file to process.

## Ollama pulling in models
To pull in the llm use the ollama pull command for both models
```
ollama pull deepseek-r1:latest
ollama pull xbai-embed-large:latest
```

## Notes

- Make sure `ollama`, `deepseek-r1`, and `mxbai-embed-large` are properly installed and configured before running the script.
- For model installation instructions, refer to their respective documentation.