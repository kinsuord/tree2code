const fs = require('fs');
const os = require('os');
const path = require('path');
const puppeteer = require('puppeteer');

const infile = process.argv[2];
const outfile = process.argv[3];
const template = process.argv[4];
const isFolder = (infile.split(".").length===1) ? true : false;
const tmpDirName = "html";
// const tmpDir = path.join(os.tmpdir(), tmpDirName);
const tmpDir = tmpDirName;

if (!fs.existsSync(tmpDir)){
    fs.mkdirSync(tmpDir);
}

const xmlToHtml = (inFile, OutFile, template) => {
    const data = fs.readFileSync(inFile, 'utf8');
    let templ = fs.readFileSync(template, 'utf8');
    templ = templ.split('<div id="app">');
    templ.splice(1,0,'<div id="app">', data);
    fs.writeFileSync(OutFile, templ.join(''));
};

const getScreenShot = (inFile, OutFile) => {

};

let inputFiles=[], outputFiles=[];

// to html
if(isFolder){
    inputFiles = fs.readdirSync(infile);
    for(let i=0; i<inputFiles.length; i++){
        outputFiles.push(path.join(tmpDir, inputFiles[i].replace('.xml', '.html')));
        inputFiles[i] = path.join(infile, inputFiles[i]);
    }
}
else{
    inputFiles.push(infile);
    outputFiles.push(path.join(tmpDir, infile.replace('.xml', '.html')));
}


for(let i=0; i<inputFiles.length; i++){
    xmlToHtml(inputFiles[i], outputFiles[i], template);
    inputFiles[i] = outputFiles[i];
    if(isFolder){
        let filenames = fs.readdirSync(infile);
        outputFiles[i] = path.join(outfile, filenames[i].replace('.xml', '.jpg'))
    }
    else{
        outputFiles[i] = outfile;
    }
}

// to jpg
(async () => {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    await page.setViewport({
        width: 1280,
        height: 960,
        deviceScaleFactor: 1,
    });
    for(let i=0; i<inputFiles.length; i++){
        await page.goto('file:///' + path.resolve(inputFiles[i]));
        await page.screenshot({path: outputFiles[i]});
    }
  
    await browser.close();
})();