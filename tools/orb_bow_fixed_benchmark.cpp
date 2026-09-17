#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ORBextractor.h"
#include "ORBVocabulary.h"

using Clock = std::chrono::steady_clock;

struct Item { int frame; std::string path; };

static std::vector<Item> read_manifest(const std::string &path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Cannot open manifest: " + path);
    std::vector<Item> rows;
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty()) continue;
        const auto tab = line.find('\t');
        if (tab == std::string::npos) throw std::runtime_error("Bad manifest row");
        rows.push_back({std::stoi(line.substr(0, tab)), line.substr(tab + 1)});
    }
    return rows;
}

static std::vector<cv::Mat> descriptor_vector(const cv::Mat &descriptors) {
    std::vector<cv::Mat> result;
    result.reserve(descriptors.rows);
    for (int row = 0; row < descriptors.rows; ++row)
        result.push_back(descriptors.row(row));
    return result;
}

static DBoW2::BowVector make_bow(
    const cv::Mat &decoded,
    ORB_SLAM3::ORBextractor &extractor,
    const ORB_SLAM3::ORBVocabulary &vocabulary,
    double *elapsed_ms = nullptr
) {
    const auto started = Clock::now();
    cv::Mat gray;
    if (decoded.channels() == 3)
        cv::cvtColor(decoded, gray, cv::COLOR_BGR2GRAY);
    else
        gray = decoded;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<int> overlap = {0, gray.cols};
    extractor(gray, cv::Mat(), keypoints, descriptors, overlap);
    DBoW2::BowVector bow;
    if (!descriptors.empty())
        vocabulary.transform(descriptor_vector(descriptors), bow);
    if (elapsed_ms) {
        *elapsed_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - started).count();
    }
    return bow;
}

int main(int argc, char **argv) {
    if (argc != 7) {
        std::cerr << "usage: orb_bow_fixed_benchmark VOCAB DB.tsv QUERY.tsv "
                     "QUERY_OUT.csv LATENCY_OUT.csv REPS\n";
        return 2;
    }
    ORB_SLAM3::ORBVocabulary vocabulary;
    if (!vocabulary.loadFromTextFile(argv[1]))
        throw std::runtime_error("Could not load ORB vocabulary");
    ORB_SLAM3::ORBextractor extractor(2000, 1.2f, 8, 20, 7);
    const auto database = read_manifest(argv[2]);
    const auto queries = read_manifest(argv[3]);
    const int repetitions = std::stoi(argv[6]);

    std::unordered_map<int, DBoW2::BowVector> bows;
    for (const auto &item : database) {
        cv::Mat image = cv::imread(item.path, cv::IMREAD_UNCHANGED);
        if (image.empty()) throw std::runtime_error("Cannot read " + item.path);
        bows.emplace(item.frame, make_bow(image, extractor, vocabulary));
    }

    std::ofstream results(argv[4]);
    results << "query_frame,rank1_frame,rank1_score,rank2_frame,rank2_score,"
               "rank3_frame,rank3_score,rank4_frame,rank4_score,rank5_frame,rank5_score\n";
    for (const auto &query : queries) {
        cv::Mat image = cv::imread(query.path, cv::IMREAD_UNCHANGED);
        if (image.empty()) throw std::runtime_error("Cannot read " + query.path);
        const auto query_bow = make_bow(image, extractor, vocabulary);
        std::vector<std::pair<double, int>> scores;
        for (const auto &candidate : database) {
            if (candidate.frame > query.frame - 100) continue;
            scores.emplace_back(vocabulary.score(query_bow, bows.at(candidate.frame)),
                                candidate.frame);
        }
        std::partial_sort(scores.begin(), scores.begin() + std::min<size_t>(5, scores.size()),
                          scores.end(), std::greater<std::pair<double, int>>());
        results << query.frame;
        for (size_t rank = 0; rank < 5; ++rank) {
            if (rank < scores.size()) results << ',' << scores[rank].second << ',' << scores[rank].first;
            else results << ",,";
        }
        results << '\n';
    }

    std::ofstream latency(argv[5]);
    latency << "repetition,query_frame,descriptor_ms,search_ms,end_to_end_ms\n";
    const size_t timed_count = std::min<size_t>(30, queries.size());
    for (int repetition = 1; repetition <= repetitions; ++repetition) {
        for (size_t index = 0; index < timed_count; ++index) {
            const size_t selected = timed_count == 1 ? 0 :
                index * (queries.size() - 1) / (timed_count - 1);
            const auto &query = queries[selected];
            cv::Mat image = cv::imread(query.path, cv::IMREAD_UNCHANGED);
            double descriptor_ms = 0.0;
            const auto query_bow = make_bow(image, extractor, vocabulary, &descriptor_ms);
            const auto search_started = Clock::now();
            std::vector<std::pair<double, int>> scores;
            for (const auto &candidate : database) {
                if (candidate.frame > query.frame - 100) continue;
                scores.emplace_back(vocabulary.score(query_bow, bows.at(candidate.frame)),
                                    candidate.frame);
            }
            std::partial_sort(scores.begin(), scores.begin() + std::min<size_t>(5, scores.size()),
                              scores.end(), std::greater<std::pair<double, int>>());
            const double search_ms = std::chrono::duration<double, std::milli>(
                Clock::now() - search_started).count();
            latency << repetition << ',' << query.frame << ',' << descriptor_ms << ','
                    << search_ms << ',' << descriptor_ms + search_ms << '\n';
        }
    }
    return 0;
}
