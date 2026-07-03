#define AppName "DeepLC"
#define AppPublisher "CompOmics"
#define AppURL "https://github.com/compomics/DeepLC"
#define AppExeName "deeplc.exe"

[Setup]
AppId={{5540C6D9-E2DE-42EC-90A7-8598F55EA165}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DisableProgramGroupPage=yes
LicenseFile=.\LICENSE
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
SetupIconFile=.\img\deeplc.ico
OutputDir="dist"
OutputBaseFilename="{#AppName}-{#AppVersion}-Windows64bit"
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist\deeplc\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Parameters: "gui --native"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Parameters: "gui --native"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Parameters: "gui --native"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
