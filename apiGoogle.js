const textToSpeech = require("@google-cloud/text-to-speech");
const fs = require("fs");
const util = require("util");

async function performTextToSpeech(inputText) {
  //creating client
  const client = new textToSpeech.TextToSpeechClient();
  async function quickStart() {
    // the text to synthesize EMPTY AT FIRST
    const text = inputText;
  }

  // Construct the request
  const request = {
    input: { text: text },
    // Select the language and SSML voice gender (optional)
    voice: { languageCode: "en-US", ssmlGender: "NEUTRAL" },
    // select the type of audio encoding
    audioConfig: { audioEncoding: "MP3" },
  };

  try {
    // performs the api request
    // performs the text-to-speech request
    const [response] = await client.sythesizeSpeech(request);

    // writing the binary autio
    const writeFile = util.promisify(fs.write);
    await writeFile("output.mp3", response.audioContent, "binary");
    console.log("Audio content written");
  } catch (error) {
    console.error("Error:", error);
  }
}
