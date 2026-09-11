const results = [
    // AI
    { category: "AI", actual: "AI", predicted: "AI" },
    { category: "AI", actual: "AI", predicted: "AI" },
    { category: "AI", actual: "AI", predicted: "AI" },
    { category: "AI", actual: "AI", predicted: "AI" },
    { category: "AI", actual: "AI", predicted: "AI" },

    // E-COMMERCE
    { category: "E-COMMERCE", actual: "REAL", predicted: "REAL" },
    { category: "E-COMMERCE", actual: "REAL", predicted: "REAL" },
    { category: "E-COMMERCE", actual: "REAL", predicted: "REAL" },
    { category: "E-COMMERCE", actual: "REAL", predicted: "REAL" },
    { category: "E-COMMERCE", actual: "REAL", predicted: "REAL" },

    // EDITED
    { category: "EDITED", actual: "REAL", predicted: "AI" },
    { category: "EDITED", actual: "REAL", predicted: "REAL" },
    { category: "EDITED", actual: "REAL", predicted: "REAL" },
    { category: "EDITED", actual: "REAL", predicted: "REAL" },
    { category: "EDITED", actual: "REAL", predicted: "AI" },

    // ORIGINAL
    { category: "ORIGINAL", actual: "REAL", predicted: "REAL" },
    { category: "ORIGINAL", actual: "REAL", predicted: "AI" },
    { category: "ORIGINAL", actual: "REAL", predicted: "REAL" },
    { category: "ORIGINAL", actual: "REAL", predicted: "REAL" },
    { category: "ORIGINAL", actual: "REAL", predicted: "REAL" }
];

let TP = 0;
let TN = 0;
let FP = 0;
let FN = 0;

for (const r of results) {

    if (r.actual === "AI" && r.predicted === "AI") {
        TP++;
    }

    else if (r.actual === "REAL" && r.predicted === "REAL") {
        TN++;
    }

    else if (r.actual === "REAL" && r.predicted === "AI") {
        FP++;
    }

    else if (r.actual === "AI" && r.predicted === "REAL") {
        FN++;
    }
}

const total = results.length;

const accuracy = (TP + TN) / total;

const precision =
    TP / (TP + FP);

const recall =
    TP / (TP + FN);

const f1 =
    2 * precision * recall /
    (precision + recall);

const falsePositiveRate =
    FP / (FP + TN);

const falseNegativeRate =
    FN / (FN + TP);

console.log();
console.log("=".repeat(70));
console.log("IMAGEGUARD - 20 IMAGE CLASSIFICATION METRICS");
console.log("=".repeat(70));

console.log();
console.log("CONFUSION MATRIX");
console.log("-".repeat(70));

console.log("                 Predicted REAL    Predicted AI");
console.log(
    `Actual REAL          ${TN}              ${FP}`
);

console.log(
    `Actual AI            ${FN}              ${TP}`
);

console.log();
console.log("METRICS");
console.log("-".repeat(70));

console.log(
    `Accuracy            : ${(accuracy * 100).toFixed(2)}%`
);

console.log(
    `Precision           : ${(precision * 100).toFixed(2)}%`
);

console.log(
    `Recall              : ${(recall * 100).toFixed(2)}%`
);

console.log(
    `F1 Score            : ${(f1 * 100).toFixed(2)}%`
);

console.log(
    `False Positive Rate : ${(falsePositiveRate * 100).toFixed(2)}%`
);

console.log(
    `False Negative Rate : ${(falseNegativeRate * 100).toFixed(2)}%`
);

console.log();
console.log("ERROR CASES");
console.log("-".repeat(70));

for (const r of results) {

    if (r.actual !== r.predicted) {

        console.log(
            `${r.category.padEnd(15)} Actual: ${r.actual.padEnd(5)} Predicted: ${r.predicted}`
        );

    }
}

console.log();
console.log("=".repeat(70));
console.log("EVALUATION COMPLETE");
console.log("=".repeat(70));