Unicode true
RequestExecutionLevel user
SilentInstall silent
AutoCloseWindow true
InstallDir "$TEMP\TyloInferStudio"
SetCompressor /SOLID zlib
Name "TyloInfer"
Caption "TyloInfer"
OutFile "@OUTPUT@"
Icon "@ICON@"
VIProductVersion "@VERSION@.0"
VIAddVersionKey /LANG=1033 "ProductName" "TyloInfer"
VIAddVersionKey /LANG=1033 "FileDescription" "TyloInfer"
VIAddVersionKey /LANG=1033 "FileVersion" "@VERSION@"
VIAddVersionKey /LANG=1033 "LegalCopyright" "MFQ contributors"

!include "FileFunc.nsh"
!include "LogicLib.nsh"

Var Parameters
Var CacheRoot
Var CachePath
Var Stage
Var StagingRoot
Var Mutex
Var WaitPid

Section
  StrCpy $Mutex 0
  ${GetParameters} $Parameters
  ReadEnvStr $CacheRoot "TYLOINFER_STUDIO_RUNTIME_CACHE"
  ${If} $CacheRoot == ""
    StrCpy $CacheRoot "$LOCALAPPDATA\com.tylogi.tyloinfer\runtime"
  ${EndIf}
  ${GetOptions} $Parameters "/CACHE_DIR=" $0
  ${IfNot} ${Errors}
    CreateDirectory "$0"
    IfErrors failed
    GetFullPathName $CacheRoot "$0"
    IfErrors failed
  ${EndIf}
  StrCpy $CachePath "$CacheRoot\@PAYLOAD_ID@"

  System::Call 'kernel32::CreateMutexW(p 0, i 0, w "Local\TyloInferStudio-@PAYLOAD_ID@") p .r0'
  StrCpy $Mutex $0
  ${If} $Mutex == 0
    Goto failed
  ${EndIf}
  System::Call 'kernel32::WaitForSingleObject(p $Mutex, i 120000) i .r0'
  ${If} $0 != 0
  ${AndIf} $0 != 128
    Goto failed
  ${EndIf}

  IfFileExists "$CachePath\.ready" 0 extract
@PAYLOAD_CHECKS@
  Goto cached

extract:
  ClearErrors
  SetOutPath "$CacheRoot"
  IfErrors failed
  GetTempFileName $StagingRoot "$CacheRoot"
  IfErrors failed
  ; 只删除刚预留的单个空文件，旧缓存和失败载荷留待手动清理。
  Delete "$StagingRoot"
  IfErrors failed
  CreateDirectory "$StagingRoot"
  IfErrors failed
  StrCpy $Stage "$StagingRoot\payload"
  ClearErrors
@PAYLOAD_FILES@
  FileOpen $0 "$Stage\.ready" w
  IfErrors failed
  FileWrite $0 "@PAYLOAD_ID@"
  FileClose $0
  IfErrors failed
  SetOutPath "$CacheRoot"
  IfFileExists "$CachePath\." 0 publish
  Rename "$CachePath" "$StagingRoot\previous"
  IfErrors failed
publish:
  ClearErrors
  Rename "$Stage" "$CachePath"
  IfErrors rollback
  Goto cached

rollback:
  IfFileExists "$StagingRoot\previous\." 0 failed
  Rename "$StagingRoot\previous" "$CachePath"
  Goto failed

cached:
  System::Call 'kernel32::ReleaseMutex(p $Mutex)'
  System::Call 'kernel32::CloseHandle(p $Mutex)'
  StrCpy $Mutex 0
  ClearErrors
  ${GetOptions} $Parameters "/EXTRACT_ONLY" $0
  ${IfNot} ${Errors}
    Goto done
  ${EndIf}

  ClearErrors
  ${GetOptions} $Parameters "/WAITPID=" $WaitPid
  ${IfNot} ${Errors}
    System::Call 'kernel32::OpenProcess(i 0x100000, i 0, i $WaitPid) p .r0'
    ${If} $0 != 0
      StrCpy $1 $0
      System::Call 'kernel32::WaitForSingleObject(p r1, i 30000) i .r0'
      System::Call 'kernel32::CloseHandle(p r1)'
      ${If} $0 != 0
        Goto failed
      ${EndIf}
    ${EndIf}
  ${EndIf}

  System::Call 'kernel32::SetEnvironmentVariableW(w "TYLOINFER_STUDIO_LAUNCHER", w "$EXEPATH")'
  SetOutPath "$CachePath"
  ClearErrors
  Exec '"$CachePath\tyloinfer.exe"'
  IfErrors failed

done:
  SetErrorLevel 0
  Quit

failed:
  ${If} $Mutex != 0
    System::Call 'kernel32::ReleaseMutex(p $Mutex)'
    System::Call 'kernel32::CloseHandle(p $Mutex)'
  ${EndIf}
  MessageBox MB_OK|MB_ICONSTOP "TyloInfer could not start. Check available disk space and write access to:$\r$\n$CacheRoot"
  SetErrorLevel 1
SectionEnd
