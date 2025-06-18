import modal, json, tempfile, subprocess, uuid, os
from fastapi import File, UploadFile, Form, Header
from fastapi.responses import JSONResponse

image = (
    modal.Image.micromamba(python_version="3.10")
        .micromamba_install(
            "montreal-forced-aligner",
            "fastapi",
            "uvicorn",
            "python-dotenv",
            "requests",
            channels=["conda-forge"],
        )
        .run_commands(
            "apt-get update && apt-get install -y libquadmath0",
            "mfa model download acoustic portuguese_mfa",
            "mfa model download dictionary portuguese_brazil_mfa",
        )
)


app = modal.App("modal-mfa")

@app.cls(gpu=None, image=image, timeout=120, secrets=[modal.Secret.from_name("mfa-secret")])
class Model:
    def inference(self, text: str, audio_bytes: bytes):
        tmp = tempfile.mkdtemp()
        stem = uuid.uuid4().hex
        wav = f"{tmp}/{stem}.wav"
        lab = f"{tmp}/{stem}.lab"
        out = f"{tmp}/aligned"
        open(wav, "wb").write(audio_bytes)
        open(lab, "w", encoding="utf8").write(text)
        subprocess.run(
            [
                "mfa",
                "align",
                tmp,
                "portuguese_brazil_mfa",
                "portuguese_mfa",
                out,
                "--output_format",
                "json",
                "--clean",
            ],
            check=True,
        )
        return json.load(open(f"{out}/{stem}.json", encoding="utf8"))

    @modal.method()
    def _inference(self, text: str, audio_file: bytes):
        return self.inference(text, audio_file)

    @modal.fastapi_endpoint(docs=True, method="POST")
    def web_inference(
        self,
        text: str = Form(...),
        audio_file: UploadFile = File(...),
        x_api_key: str = Header(None),
    ):
        if x_api_key != os.getenv("API_KEY"):
            return JSONResponse(status_code=401, content={"message": "Unauthorized"})
        return JSONResponse(content=self.inference(text, audio_file.file.read()))

@app.local_entrypoint()
def main():
    audio = open("example.wav", "rb").read()
    res = Model()._inference.remote("Olá, tudo bem?", audio)
    print(json.dumps(res, indent=2, ensure_ascii=False))
