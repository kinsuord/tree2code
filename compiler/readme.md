# generate dataset

## Content

- dsl2xml.py: dsl to xml
- screenshot.js: xml to jpg using vue component and puppeteer

## Requirement

- python
- node.js
- puppeteer

## command

### dsl to xml

see the file in dsl2xml.py. you can see some setting there.

```python
# line 4
source_dir = os.path.join('dataset','gui')
target_dir = os.path.join('dataset','pure_tag')
mapping_file = os.path.join('dataset', 'pruetag_mapping.json')
```

### xml to png

```cmd
node screenshot.js [input] [output] [template]
```

- input: can be file or folder
- output: the same type as input
- template: .html for vue to compile

ex

```cmd
node screenshot.js dataset\pure_tag dataset\image template.html

node screenshot.js 0.xml out.png template.html
```
