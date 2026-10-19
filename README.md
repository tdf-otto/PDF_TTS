# PDF_TTS
A little program to generate audio files of class readings from pdfs, to listen on the go. Runs on Surya OCR and Kokoro.


## Step-By-Step for MacOS

> Commands Quick Reference
> ```
> cd ./Documents/PDF_TTS
> pixi install
> pixi run pdf
> ```
> OR `pixi run txt`
> 

### 0. Setup
Open the 'Terminal' application and run

`brew install pixi`


Either:
- Install the PDF_TTS folder somewhere on your computer, e.g. in your Documents folder. 
- or run 
  `brew install git` 
  and then 
  `git clone https://github.com/tdf-otto/PDF_TTS.git` 
  for the most up-to-date version. (Hopefully works)
### 1. Run program

Either:
- copy the text you want read aloud into a .txt file (e.g. with program 'TextEdit'), which makes it easier for you to control exactly what the text is that is read aloud. Save the file to the /texts folder in PDF_TTS. In 3.,  proceed with `pixi run txt`.
- save the pdf you want to have read aloud on your computer (extra clean if it's in a PDFs folder in the PDF_TTS folder, but can also work e.g. from your Obsidian folder structure). This is faster, but means you may get stuff like page numbers and links read out. Proceed with `pixi run pdf`.


**In your Terminal, run**
1. `cd ./Documents/PDF_TTS` *(or adjust the file path if the folder is somewhere else)*
2. `pixi install`
3. `pixi run txt` OR `pixi run pdf`, depending on what you need

Follow the instructions in the command line. 


- Running the TXT file requires you to copy the filename in the texts folder exactly. This is easiest with right-click > Rename.
- Running PDF for the first time will require some larger downloads.
- When running PDF, there will be a popup Finder window which you may open in the background. If it seems like the program is sitting around doing nothing, look for it, e.g. by finding "python3.12" in the dock

### 2. Happy listening
