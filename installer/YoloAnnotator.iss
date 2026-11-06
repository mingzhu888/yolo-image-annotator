#define MyAppName "云企智瞰标注工具"
#define MyAppVersion "1.2.1"
#define MyAppExeName "云企智瞰标注工具.exe"

[Setup]
AppId={{7B879A8D-43E0-49B8-82F8-29AF3155E845}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=mingzhu888
DefaultDirName={localappdata}\Programs\云企智瞰标注工具
DefaultGroupName=云企智瞰标注工具
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=output
OutputBaseFilename=云企智瞰标注工具_安装程序_1.2.1
Compression=lzma2/fast
SolidCompression=no
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
SetupIconFile=..\assets\yunqi-zhikan-annotator.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
AppMutex=Local\YoloImageAnnotatorDesktop
SetupLogging=yes
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加任务："; Flags: checkedonce

[Files]
Source: "..\dist\云企智瞰标注工具\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "redist\MicrosoftEdgeWebview2Setup.exe"; DestDir: "{tmp}"; Flags: dontcopy deleteafterinstall
Source: "redist\vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: dontcopy deleteafterinstall

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent

[Code]
function IsWebView2Installed(): Boolean;
var
  Version: String;
begin
  Result :=
    RegQueryStringValue(HKCU, 'Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) or
    RegQueryStringValue(HKLM32, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) or
    RegQueryStringValue(HKLM64, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version);
end;

function IsVCRuntimeInstalled(): Boolean;
var
  Installed: Cardinal;
begin
  Result :=
    (RegQueryDWordValue(HKLM64, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Installed', Installed) and (Installed = 1)) or
    (RegQueryDWordValue(HKLM32, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Installed', Installed) and (Installed = 1));
end;

procedure RunPrerequisite(FileName, Parameters, Description: String);
var
  ResultCode: Integer;
  FilePath: String;
begin
  ExtractTemporaryFile(FileName);
  FilePath := ExpandConstant('{tmp}\' + FileName);
  if (not Exec(FilePath, Parameters, '', SW_SHOW, ewWaitUntilTerminated, ResultCode)) or (ResultCode <> 0) then
    MsgBox(Description + ' 安装失败，错误代码：' + IntToStr(ResultCode) + #13#10 +
           '应用仍会继续安装；如果程序无法启动，请联系管理员安装该组件。',
           mbError, MB_OK);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    if not IsVCRuntimeInstalled() then
      RunPrerequisite('vc_redist.x64.exe', '/install /quiet /norestart', 'Microsoft Visual C++ 运行库');
    if not IsWebView2Installed() then
      RunPrerequisite('MicrosoftEdgeWebview2Setup.exe', '/silent /install', 'Microsoft Edge WebView2 Runtime');
  end;
end;
