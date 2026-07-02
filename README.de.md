# ComfyUI Model Search and Load

<p align="center">
  <img src="assets/logo.svg" alt="ComfyUI Model Search and Load Logo" width="128" height="128">
</p>

Eine ComfyUI-Erweiterung, die einen geladenen Workflow nach fehlenden
Modell-Dateien durchsucht, sie auf HuggingFace und CivitAI findet und
direkt in den korrekten Ordner herunterlädt. Spart Speicherplatz indem
bereits vorhandene Kopien per Hardlink wiederverwendet werden können.

> [English version: README.md](README.md)

---

> ## ⚠️ Haftungsausschluss
>
> **Dieses Projekt ist "vibe-coded".** Es wurde iterativ in einem
> Dialog mit einem KI-Coding-Assistenten entwickelt - veröffentlicht
> weil es beim Autor funktioniert, nicht weil jeder Code-Pfad Zeile für
> Zeile geprüft wurde.
>
> **Nutzung auf eigenes Risiko.** Der Autor stellt diese Software OHNE
> jegliche Gewährleistung bereit. Der Autor übernimmt **keinerlei
> Haftung** für irgendetwas das aus der Nutzung dieses Codes entsteht,
> insbesondere nicht für: gelöschte oder beschädigte Dateien, Downloads
> an falscher Stelle, vollgelaufene Festplatten, geleakte API-Tokens,
> beschädigte ComfyUI-Installationen, unerwartete Bandbreiten- oder
> Speicherkosten, oder sonstige direkte oder indirekte Schäden.
>
> Das Plugin verschiebt Dateien, löscht `.part`-Dateien unvollständiger
> Downloads, erstellt Hardlinks/Symlinks im gesamten `models/`-Baum und
> speichert API-Tokens in einer lokalen `config.json`. **Sichere alles
> was du nicht verlieren kannst, bevor du es zum ersten Mal startest.**
> Wenn möglich, erst auf einer Test-ComfyUI-Installation ausprobieren.
>
> Die formalen Bedingungen stehen in der [LICENSE](LICENSE)-Datei (MIT).

---

## Features

- **Workflow-Scan auf Knopfdruck** erkennt jede referenzierte
  Modell-Datei in geladenen Workflows - quer über alle Custom-Node-Packs
  - und sagt dir welche fehlen.
- **Intelligentes Folder-Routing** nutzt ComfyUI's eigenes
  `folder_paths`, deine `extra_model_paths.yaml`, von Custom-Nodes
  registrierte Ordner (z.B. Kijai's `detection/` für ViTPose / YOLO)
  und Filename-Heuristiken um den richtigen Zielordner zu finden.
  Subfolder-Pfade aus dem Workflow (`Wan2_2/lightx2v/...`) bleiben
  erhalten.
- **Mehrstufige Suche**:
  1. Kuratierte Datenbank bekannter Modelle (FLUX, SDXL, ControlNet v1.1, ...)
  2. HuggingFace Repo-Namen-Suche
  3. HuggingFace **Volltext-README-Suche** (findet Dateien die in
     READMEs erwähnt werden, auch wenn der Repo-Name nicht zum Filename
     passt)
  4. Fallback-Liste bekannter Sammelrepos (`Kijai/WanVideo_comfy`,
     `Comfy-Org/frame_interpolation`, `lightx2v/Wan2.2-Distill-Loras`, ...)
  5. CivitAI-Suche
- **Background-Downloader** mit Fortschrittsbalken, smoother Animation,
  Geschwindigkeit, ETA, Resume-Support, Cancel und HTTP-Fehler-Erkennung
  (fängt die "Server hat eine HTML-Login-Seite statt des Modells
  geliefert"-Falle ab).
- **Speicherplatz sparen**: wenn ein Modell mit identischem Namen +
  Größe schon irgendwo in deinem `models/`-Baum existiert, kann das
  Plugin es per Hardlink (oder Symlink) verlinken statt erneut
  herunterzuladen. Optional, muss aktiv eingeschaltet werden.
- **Move existing**: verschiebt Dateien die im falschen Ordner gelandet
  sind an die korrekte Stelle - kein Re-Download nötig.
- **Duplikat-Schutz**: ein zweiter Download-Klick auf ein bereits
  laufendes Modell zeigt eine deutliche Toast-Benachrichtigung statt
  einen parallelen Transfer zu starten.
- **API-Tokens** für HuggingFace gated Repos (FLUX.1-dev) und CivitAI.
  Tokens werden in einer lokalen `config.json` gespeichert und
  **niemals** ans Frontend zurückgesendet - nur eine maskierte Vorschau
  (`hf_xx••••wxyz`) wird angezeigt.

---

## Installation

Repo ins ComfyUI-Custom-Nodes-Verzeichnis klonen (oder kopieren):

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<dein-username>/comfyui-modelsearchandload.git
```

ComfyUI neu starten. In der Sidebar erscheint ein **Models**-Tab
(Download-Icon). Auf älteren ComfyUI-Versionen ohne Sidebar-API gibt es
einen schwebenden "Models"-Button unten rechts.

Es werden keine zusätzlichen Python-Pakete benötigt - das Plugin nutzt
nur die Standardbibliothek (`urllib`, `threading`, `json`, ...) und das
bereits in ComfyUI vorhandene `aiohttp`.

### Plattformübergreifend

Getestet auf Windows und Linux. Alle Pfade nutzen `os.path.join`, alle
String-Vergleiche `os.path.normcase`, und Link-Operationen fallen
automatisch zurück wenn das OS sie nicht unterstützt
(z.B. Windows-Symlinks ohne Developer-Mode → fällt auf Hardlink zurück
→ fällt auf Copy zurück).

---

## Bedienung

1. **Workflow laden** in ComfyUI.
2. Sidebar-Tab **Models** öffnen.
3. **Scan workflow** klicken - fehlende Modelle erscheinen als Liste
   mit ihrem Ziel-Ordner.
4. **Find sources** bei einem einzelnen Eintrag klicken, Kandidaten
   prüfen und beim gewünschten Treffer **Download** drücken.
5. Fortschritt im **Downloads**-Panel beobachten. Jede Zeile zeigt
   Live-Status, Prozent, Geschwindigkeit und ETA. Fertige Files faden
   aus der Missing-Liste aus.

Weitere nützliche Aktionen:

- **Move existing** durchsucht deinen `models/`-Baum nach Dateien die
  per Name zu einem fehlenden Eintrag passen, und verschiebt sie an die
  von ComfyUI erwartete Stelle (korrekter Ordner + Subfolder).
- **Settings** öffnet einen Dialog für HuggingFace/CivitAI API-Tokens
  und die Disk-Space-Optionen (Linking).

---

## Speicherplatz: Linken statt Kopieren

Wenn dasselbe Modell aus zwei verschiedenen Pfaden referenziert wird
(z.B. `Wan2_2/foo.safetensors` und `foo.safetensors`), oder wenn ein
Download für eine Datei angefordert wird, die schon irgendwo im
`models/`-Baum liegt, kann das Plugin einen Filesystem-Link erstellen
statt die Daten erneut herunterzuladen.

**Settings → Disk space** öffnen, *Reuse existing files via filesystem
links* aktivieren und Modus wählen:

| Modus | Verhalten |
|---|---|
| **Auto** *(empfohlen)* | Erst Hardlink (sofort, kein extra Platz). Wenn das fehlschlägt (Cross-Filesystem, keine Rechte), Symlink probieren. Wenn auch Symlinks fehlschlagen (Windows ohne Developer-Mode), normaler Download. |
| **Hardlink only** | Nur Hardlinks. Schlägt fehl wenn die Datei auf einem anderen Filesystem liegt. |
| **Symlink only** | Nur Symlinks. Kann auf Windows ohne Developer-Mode oder Admin-Rechte fehlschlagen. |

Das Plugin matched über **Filename + Dateigröße**. SHA256-Verifikation
gibt es nicht weil mehrere GB pro Modell zu lesen zu langsam wäre - der
Größencheck ist in der Praxis ausreichend um verschiedene Versionen zu
unterscheiden.

Wenn das Linken klappt, zeigt der Job-Status `✓ Hardlinked` oder
`✓ Symlinked` und es werden keine Bytes heruntergeladen.

---

## API-Tokens

Manche Modelle brauchen Authentifizierung:

- **HuggingFace Token** für gated Repos wie `black-forest-labs/FLUX.1-dev`.
  Token bei https://huggingface.co/settings/tokens generieren (read-only
  reicht). Du musst außerdem auf der HuggingFace-Webseite die Lizenz
  des Modells akzeptiert haben, sonst funktioniert der Download nicht.
- **CivitAI API Key** für viele CivitAI-Downloads.
  Bei https://civitai.com/user/account holen.

**Settings** öffnen, Token einfügen, **Save** klicken. Nach dem
Speichern wird das Eingabefeld geleert und die Status-Zeile zeigt
`✓ Stored: hf_xx••••wxyz`. Der echte Wert verlässt nie den Server.
**Clear** entfernt einen gespeicherten Token wieder.

---

## HTTP-API

Das Plugin registriert diese Endpoints auf dem ComfyUI-Server. Externe
Skripte können das Plugin damit ansteuern:

| Methode | Pfad | Body |
|---|---|---|
| `POST` | `/model_downloader/scan` | `{"workflow": <api-format-prompt>}` |
| `POST` | `/model_downloader/search` | `{"filename": "...", "folder": "..."}` |
| `POST` | `/model_downloader/web_search` | `{"filename": "...", "folder": "..."}` |
| `POST` | `/model_downloader/download` | `{"url": "...", "folder": "...", "filename": "...", "subfolder": "...", "size": <bytes>}` |
| `GET`  | `/model_downloader/jobs` | - |
| `POST` | `/model_downloader/cancel` | `{"id": "<job-id>"}` |
| `POST` | `/model_downloader/clear` | `{}` |
| `POST` | `/model_downloader/relocate` | `{"items": [{"name": "...", "folder": "...", "subfolder": "..."}, ...]}` |
| `GET`  | `/model_downloader/config` | - (Tokens werden maskiert) |
| `POST` | `/model_downloader/config` | `{"huggingface_token": "...", "civitai_token": "...", "enable_linking": true, "linking_mode": "auto"}` |

Der Download-Endpoint liefert `{"job": {...}, "duplicate": true}` zurück
wenn ein Job für denselben Filename + Ziel bereits läuft - so können
Clients verhindern dass parallele Transfers gestartet werden.

---

## Eigene bekannte Modelle hinzufügen

`known_models.json` editieren um dem Plugin ein Custom-Modell beizubringen:

```json
{
  "mein_lieblings_lora.safetensors": {
    "folder": "loras",
    "url": "https://huggingface.co/user/repo/resolve/main/file.safetensors",
    "source": "huggingface",
    "size": 134217728
  }
}
```

Folder-Keys folgen der ComfyUI-Konvention: `checkpoints`, `loras`,
`vae`, `controlnet`, `clip` (= `text_encoders`), `clip_vision`,
`diffusion_models`, `upscale_models`, `embeddings`, `style_models`,
`ipadapter`, `gligen`, `hypernetworks`, `frame_interpolation`,
`detection`, ...

Pull Requests mit Ergänzungen zur kuratierten Datenbank sind willkommen.

---

## Bekannte Einschränkungen

- **HuggingFace-Suchlücken**: sehr obskure Filenames die weder in einem
  Repo-Namen noch in irgendeiner README vorkommen können nicht
  automatisch gefunden werden. Für diese: URL direkt einfügen oder das
  Modell in `known_models.json` eintragen.
- **Gated Models** benötigen sowohl einen Token *als auch* eine
  akzeptierte Lizenz auf der HuggingFace-Webseite.
- **CivitAI Rate-Limits** - viele CivitAI-Modelle schnell hintereinander
  zu downloaden kann ihr API-Rate-Limit treffen.

---

## Beitragen

PRs willkommen. Bitte:

- Vor dem Pushen `python -c "import ast, pathlib; [ast.parse(p.read_text(encoding='utf-8')) for p in pathlib.Path('.').glob('*.py')]"`
  ausführen damit alle Python-Dateien parsen.
- Die JS-Datei mit `node --check web/model_downloader.js` validieren
  (die Datei ist als reines ES-Modul geschrieben).
- `config.json` nicht committen (steht in `.gitignore`) - die Datei
  enthält installations-spezifische Tokens.

---

## Lizenz

[MIT](LICENSE)
