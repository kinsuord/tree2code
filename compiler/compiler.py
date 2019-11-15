def tree2xml(input_tree, output_path=None):
    '''
    convert output tree to xml file,
    return the converted string
    '''
    stack = []
    waiting_queue = []
    output = ""

    waiting_queue.append(input_tree)
    while waiting_queue :
        node = waiting_queue.pop(-1)
        for children in list(reversed(node.children)):
            waiting_queue.append(children)
        if node.value != 'end':
            output += "<" + node.value + ">" 
            stack.append("</" + node.value + ">")
        else:
            output += stack.pop(-1)

    if not output_path is None:
        with open(output_path,"w") as f:
            f.write(output)

    return output

def xml2html(input_xml, output_path=None):
    """
    imput_xml : xml code, string type
    """
    with open('./compiler/template.html', 'r') as f:
        templ = f.read()
        templ = templ.split('<div id="app">')
        templ = templ[0] + '<div id="app">' + input_xml + templ[1]
    if not output_path is None:
        with open(output_path,"w") as f:
            f.write(templ)
    return templ  

def html2img(input_html, output_path=None):
    """
    input_html : html code, string type
    """
    import compiler.html_to_image as h2img
    driver = h2img.Driver()
    driver.html_to_img(html_content=input_html, saveToFileName=output_path)
    