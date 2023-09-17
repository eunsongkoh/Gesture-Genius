const path = require("path");
const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");
const textToSpeech = require("@google-cloud/text-to-speech");
const fs = require("fs");

// our pseudo back-end server

// local host is 9999
// http://localhost:9999/

// link to succesful output
// http://localhost:9999/get-Audio

// link to mp3
// http://localhost:9999/audio/output_1694924441459.mp3

const app = express();
app.use(cors());
app.use(express.json());
app.options('*', cors()); 
const client = new textToSpeech.TextToSpeechClient({
  keyFilename: "geniusgesture.json",
});

const audioDirectory = "./audio";
if (!fs.existsSync(audioDirectory)) {
  fs.mkdirSync(audioDirectory);
}

const CACHED_AUDIO_FILES = new Set(fs.readdirSync(audioDirectory));

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
  console.log(req)
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

  if (CACHED_AUDIO_FILES.has(`${text}.mp3`)) {
    console.log("Audio file already exists in cache.");
    res.send(`${text}.mp3`);
    return;
  }

  try {
    const [response] = await client.synthesizeSpeech(request);
    const audioFileName = `${text}.mp3`;
    const audioFilePath = path.join(audioDirectory, audioFileName); // Save to /audio directory
    
    // Save to cache
    CACHED_AUDIO_FILES.add(audioFileName);
    
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
