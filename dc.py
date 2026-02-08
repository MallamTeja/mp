import arxiv

client = arxiv.Client()
search = arxiv.Search(
    query="cat:cs.LG",  # Machine Learning category
    max_results=10,
    sort_by=arxiv.SortCriterion.SubmittedDate
)
results = list(client.results(search))
for paper in results:
    print(f"{paper.title} by {', '.join(author.name for author in paper.authors)}")
    print(paper.summary[:200] + "...")  # Truncated abstract
    print(paper.pdf_url, paper.entry_id)
