# https://medium.com/@elle.neal_71064/mind-mapping-with-ai-an-accessible-approach-for-neurodiverse-learners-1a74767359ff


def mermaid_chart(mindmap_code):
    html_code = f"""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css">
    <div class="mermaid">{mindmap_code}</div>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({{startOnLoad:true}});</script>
    """
    return html_code


def main():
    # scrape text - client
    # generate mindmap based on text using LLM - server
    # create marmaid_chart() - server/client?
    # output png - client
    # open with local app - client
    pass


if __name__ == "__main__":
    main()
