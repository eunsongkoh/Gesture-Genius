const textToSpeech = require("@google-cloud/text-to-speech");
const fs = require("fs");
const util = require("util");

const alphabet = 'abcdefghijklmnopqrstuvwxyz';

async function performTextToSpeech(inputText) {
  //creating client
  const client = new textToSpeech.TextToSpeechClient();

  // Construct the request
  const request = {
    input: { text: inputText },
    // Select the language and SSML voice gender (optional)
    voice: { languageCode: "en-US", ssmlGender: "NEUTRAL" },
    // select the type of audio encoding
    audioConfig: { audioEncoding: "MP3" },
  };

  try {
    // performs the api request
    // performs the text-to-speech request
    const response = client.synthesizeSpeech(request).then((response) => {
      // The response's audioContent is binary.
      const audioContent = response[0].audioContent;
      console.log(audioContent);
      

      // write to file
      fs.writeFile("output.mp3", audioContent, "binary", (err) => {
        if (err) console.log("error: ", err);
        else console.log("success");
      });
    });

    console.log("Audio content written");
  } catch (error) {
    console.error("Error:", error);
  }
}

for (let i = 0; i < alphabet.length; i++) {
  performTextToSpeech(alphabet[i]);
}

