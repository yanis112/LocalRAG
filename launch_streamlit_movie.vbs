Set WScript.Shell = CreateObject("WScript.Shell")
strPath = "powershell.exe -WindowStyle Hidden -ExecutionPolicy Bypass -File """ & _
          "C:\Users\Yanis\Documents\RAG\launch_streamlit_movie.ps1"""
WScript.Shell.Run strPath, 0, false