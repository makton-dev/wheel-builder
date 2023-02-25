import github3

gh = github3.GitHub()
repo = gh.repository(owner="pytorch", repository="pytorch")

repo_tags = []
for tag in repo.tags():
    if tag.name[0] == "v":
        print(f"{tag.name}")
