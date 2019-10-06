var fs = require('fs');
var path = require('path');
var vfile = require('to-vfile');
var parse5 = require('parse5');
var fromParse5 = require('hast-util-from-parse5');
// var inspect = require('unist-util-inspect');
// var toHTML = require('hast-util-to-html')

function html2hast(inDir, outDir){

    var astCodeDir = path.join(outDir, 'code');
    var htmlCodeDir = path.join(inDir, 'code');
    var astImgDir = path.join(outDir, 'png');
    var htmlImgDir = path.join(inDir, 'png');

    if (!fs.existsSync(outDir)){
        fs.mkdirSync(outDir);
    }

    // png
    if (!fs.existsSync(astImgDir)){
        fs.mkdirSync(astImgDir);
    }
    
    fs.readdir(htmlImgDir, function(err, items) {
        for (var i=0; i<items.length; i++) {
            fs.copyFile(path.join(htmlImgDir, items[i]), path.join(astImgDir, i + '.png'), (err) => {
                if (err) throw err;
            });
        }
    });
    
    // code
    if (!fs.existsSync(astCodeDir)){
        fs.mkdirSync(astCodeDir);
    }
    
    fs.readdir(htmlCodeDir, function(err, items) {
        for (var i=0; i<items.length; i++) {
            var doc = vfile.readSync(path.join(htmlCodeDir, i + '.xml'));
            var ast = parse5.parse(String(doc));
            var hast = fromParse5(ast, doc);
            // console.log(inspect(hast));
            fs.writeFileSync(path.join(astCodeDir, i + '.json'), JSON.stringify(hast), 'utf8');
        }
    });
}

html2hast('xml', 'xml_hast');