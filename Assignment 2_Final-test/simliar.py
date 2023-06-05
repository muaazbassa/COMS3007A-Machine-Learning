def calculate_jaccard_similarity(file1_path, file2_path):
    # Read the content of the files
    with open(file1_path, 'r') as file1:
        lines1 = file1.readlines()
    with open(file2_path, 'r') as file2:
        lines2 = file2.readlines()

    # Calculate the Jaccard similarity coefficient for lines
    set1 = set(lines1)
    set2 = set(lines2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    similarity = intersection / union

    return similarity
# Example usage
file1_path = 'trainlabels.txt'  # Replace with the path to your first file
file2_path = 'testlabels.txt'  # Replace with the path to your second file

similarity_score = calculate_jaccard_similarity(file1_path, file2_path)
print("Jaccard Similarity:", similarity_score)
