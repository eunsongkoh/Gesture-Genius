/**
 * Sythesizes sample text into an .mp3 file.
 */
async function synthesize() {
  const textToSpeech = require('@google-cloud/text-to-speech');
  const fs = require('fs');
  const util = require('util');

  const client = new textToSpeech.TextToSpeechClient();

  const text = 'GOOGLE CLOUD CAN GO SCREW OFF';

  const request = {
    input: {text: text},
    voice: {languageCode: 'en-US', ssmlGender: 'NEUTRAL'},
    audioConfig: {audioEncoding: 'MP3'},
  };

  const [response] = await client.synthesizeSpeech(request);
  // Write the binary audio content to a local file
  const writeFile = util.promisify(fs.writeFile);
  await writeFile('output.mp3', response.audioContent, 'binary');
  console.log('Audio content written to file: output.mp3');
}

synthesize();