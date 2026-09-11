import fs from "fs";
import path from "path";
import { sendImageToPython } from "./services/pythonService.js";
import { calculateTrustScore } from "./services/trustScoreService.js";

const TEST_DIR = path.resolve(
    "../python-engine/test_images"
);


function getCategory(filename) {

    const name = filename.toLowerCase();

    if (name.startsWith("ai")) {
        return "AI";
    }

    if (name.startsWith("original")) {
        return "ORIGINAL";
    }

    if (name.startsWith("edited")) {
        return "EDITED";
    }

    if (name.startsWith("e-commerce")) {
        return "E-COMMERCE";
    }

    return "UNKNOWN";
}


async function main() {

    console.log("=".repeat(100));
    console.log("IMAGEGUARD - 20 IMAGE BASELINE TEST");
    console.log("=".repeat(100));

    if (!fs.existsSync(TEST_DIR)) {

        console.error(
            "\nTest image folder not found:"
        );

        console.error(TEST_DIR);

        return;
    }


    const files = fs.readdirSync(TEST_DIR)
        .filter(file =>
            /\.(jpg|jpeg|png)$/i.test(file)
        )
        .sort();


    console.log(`\nTotal images found: ${files.length}\n`);


    const results = [];


    for (const filename of files) {

        const category = getCategory(filename);

        const imagePath = path.join(
            TEST_DIR,
            filename
        );


        console.log("-".repeat(100));

        console.log(
            `Testing: ${filename} | Category: ${category}`
        );


        try {

            // -----------------------------------------
            // Python Engine
            // -----------------------------------------

            const pythonAnalysis =
                await sendImageToPython(
                    imagePath,
                    category
                );


            // -----------------------------------------
            // Existing Trust Score
            // -----------------------------------------

            const trustResult =
                calculateTrustScore({

                    aiDetection:
                        pythonAnalysis.aiDetection,

                    metadataAnalysis:
                        pythonAnalysis.metadataAnalysis

                });


            const result = {

                filename,

                category,

                prediction:
                    pythonAnalysis.aiDetection.prediction,

                aiProbability:
                    pythonAnalysis.aiDetection.ai_probability,

                confidence:
                    pythonAnalysis.aiDetection.confidence,

                metadataIntegrity:
                    pythonAnalysis.metadataAnalysis.integrityScore,

                trustScore:
                    trustResult.trustScore,

                riskLevel:
                    trustResult.riskLevel,

                recommendation:
                    trustResult.recommendation

            };


            results.push(result);


            console.log(
                `Prediction : ${result.prediction}`
            );

            console.log(
                `AI Score   : ${(result.aiProbability * 100).toFixed(2)}%`
            );

            console.log(
                `Metadata   : ${result.metadataIntegrity}/100`
            );

            console.log(
                `Trust Score: ${result.trustScore}/100`
            );

            console.log(
                `Risk       : ${result.riskLevel}`
            );


        } catch (error) {

            console.error(
                `ERROR: ${error.message}`
            );

            results.push({

                filename,

                category,

                prediction: "ERROR",

                aiProbability: null,

                confidence: null,

                metadataIntegrity: null,

                trustScore: null,

                riskLevel: "ERROR",

                recommendation: error.message

            });

        }

    }


    // =============================================
    // FINAL SUMMARY
    // =============================================

    console.log("\n\n");

    console.log("=".repeat(120));
    console.log("FINAL 20-IMAGE RESULTS");
    console.log("=".repeat(120));


    console.log(
        "FILE".padEnd(22) +
        "CATEGORY".padEnd(14) +
        "PREDICTION".padEnd(18) +
        "AI %".padEnd(10) +
        "TRUST".padEnd(10) +
        "RISK"
    );


    console.log("-".repeat(120));


    for (const result of results) {

        console.log(

            result.filename.padEnd(22) +

            result.category.padEnd(14) +

            result.prediction.padEnd(18) +

            (
                result.aiProbability !== null
                    ? `${(result.aiProbability * 100).toFixed(2)}%`
                    : "ERROR"
            ).padEnd(10) +

            (
                result.trustScore !== null
                    ? `${result.trustScore}`
                    : "ERROR"
            ).padEnd(10) +

            result.riskLevel

        );

    }


    // =============================================
    // CATEGORY SUMMARY
    // =============================================

    console.log("\n");

    console.log("=".repeat(100));
    console.log("CATEGORY SUMMARY");
    console.log("=".repeat(100));


    const categories = [
        "AI",
        "ORIGINAL",
        "EDITED",
        "E-COMMERCE"
    ];


    for (const category of categories) {

        const categoryResults =
            results.filter(
                result =>
                    result.category === category &&
                    result.trustScore !== null
            );


        if (categoryResults.length === 0) {
            continue;
        }


        const averageTrust =
            categoryResults.reduce(
                (sum, result) =>
                    sum + result.trustScore,
                0
            ) / categoryResults.length;


        const averageAI =
            categoryResults.reduce(
                (sum, result) =>
                    sum + result.aiProbability,
                0
            ) / categoryResults.length;


        console.log(
            `\n${category}`
        );

        console.log(
            `Images           : ${categoryResults.length}`
        );

        console.log(
            `Average AI Score : ${(averageAI * 100).toFixed(2)}%`
        );

        console.log(
            `Average Trust    : ${averageTrust.toFixed(2)}/100`
        );

    }


    console.log("\n");

    console.log("=".repeat(100));
    console.log("20-IMAGE TEST COMPLETE");
    console.log("=".repeat(100));

}


main();