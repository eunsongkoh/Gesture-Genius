const path = require("path");
const express = require("express");
const bodyParser = require("body-parser");
const textToSpeech = require("@google-cloud/text-to-speech");
const fs = require("fs");

const app = express();
const client = new textToSpeech.TextToSpeechClient({
  keyFilename: "geniusgesture.json",
});

const audioDirectory = "./audio";
if (!fs.existsSync(audioDirectory)) {
  fs.mkdirSync(audioDirectory);
}

app.use(bodyParser.urlencoded({ extended: true }));

app.get("/", (req, res) => {
  res.send(`
        <form action="/get-Audio" method="post">
            <textarea name="text" rows="4" cols="50" placeholder="Enter your text here..."></textarea><br>
            <input type="submit" value="Convert to Audio">
        </form>
    `);
});

app.use("/audio", express.static(audioDirectory));

app.post("/get-Audio", async (req, res) => {
  const text = req.body.text;
  const request = {
    input: { text: text },
    voice: {
      languageCode: "en-US",
      ssmlGender: "NEUTRAL",
    },
    audioConfig: {
      audioEncoding: "MP3",
    },
  };

  try {
    const [response] = await client.synthesizeSpeech(request);
    const audioFileName = `output_${Date.now()}.mp3`;
    const audioFilePath = path.join(audioDirectory, audioFileName); // Save to /audio directory
    fs.writeFileSync(audioFilePath, response.audioContent, "binary");
    console.log(`Audio content written to file: ${audioFilePath}`);
    res.send(audioFileName); // Just send back the filename
  } catch (error) {
    console.error("Error while generating audio:", error);
    res.status(500).send("Failed to generate audio.");
  }
});

const PORT = 9999;
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
