def roc(evaluation):
    allRel = []
    totalScore = 0
    totalAccuracy = 0
    for e in evaluation:
        allRel.append(e[0])
        allRel.append(e[1])
    allRel = list(set(allRel))

    for i, rel in enumerate(allRel):
        true = []
        score = []
        for sample in evaluation:
            true.append(1 if sample[0] == rel else 0)
            score.append(1 if sample[1] == rel else 0)
        try:
            totalScore += roc_auc_score(np.array(true), np.array(score))
            totalAccuracy += accuracy_score(np.array(true), np.array(score))
        except:
            # print true
            # print score
            continue

    count = 0
    for sample in evaluation:
        if sample[0] == sample[1]:
            count += 1
    print "Count:", count, " Evaluation: ", len(evaluation)
    return totalScore * 1.0 / len(allRel), totalAccuracy * 1.0 / len(allRel)
