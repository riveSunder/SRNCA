import os


def test_commit(): #pragma: no cover

    run_tests_command = "coverage run -m testing.test_all"
    make_html_command = "coverage html"
    make_report_command = "coverage report > coverage.txt"

    os.system(run_tests_command)
    os.system(make_html_command)
    os.system(make_report_command)

    with open("coverage.txt", "r") as f:
        for line in f.readlines():
            if "TOTAL" in line:
                coverage_summary = line


    
    if 0: # not writing to README for now
        if os.path.exists("README.md"):
            with open("README.md", "r") as f:
                lines = f.readlines()

            with open("README.md", "a") as f:
                f.writelines(coverage_summary)
        else:
            print("no README.md found")

    git_add_command = "git add coverage.txt"
    commit_command = f"git commit -m 'test commit summary: {coverage_summary}'"

    os.system(git_add_command)
    os.system(commit_command)

if __name__ == "__main__": #pragma: no cover
    test_commit()

