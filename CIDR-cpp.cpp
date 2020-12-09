// draft using Armadillo and mlpack libraries

#include <stdio.h>
#include <math.h>
// Include mlpack components
#include <mlpack/core.hpp>
#include <mlpack/methods/kde/kde.hpp>

#include "scPCA.hpp"
#include "kde.h"
// #include <minpack.h>

// #include <scPCA.h>
//can probably just edit the file to have a header and cpp code

using namespace arma;
using namespace mlpack;
using namespace mlpack::kde;

int main(void)
{

  // Load data using mlpack
  arma::mat data;
  data::Load("GSE131907_raw_UMI_matrix_3kcell.csv", data, false, false, arma::csv_ascii);

  // Since Armadillo transposes the matrix after loading, the number of samples is the number of rows, not columns as in the R CIDR implementation.
  int sampleSize = data.n_cols ;
  int tagSize = data.n_rows;

  // Checking if any tags aren't expressed in any cells, removing those tags
  colvec rowSums = sum(data, 1);
  for (int i = 0; i < tagSize; i++)  {
    if (rowSums(i) == 0)
    {
      data.shed_row(i);
    }
  }

  sampleSize = data.n_cols;
  tagSize = data.n_rows;
  rowvec librarySizes = sum(data, 0);

  // Create matrix that keeps track of log tags per million
  arma::mat nData = (data.t()/librarySizes).t();
  nData = nData * 1000000 + 1;
  nData.for_each([] (mat::elem_type& val) { val = log2(val); });


  /*
    Determine Dropout Candidates

    ** need to check transpose and stuff and make sure it's okay
  */
  int N = 2000;
  // should be 3000 or whatever N is
  // Want to order librarySizes and then grab the first N of those indices
  // Use sort_index in arma library
  arma::vec dTs = zeros<vec>(sampleSize);

  arma::uvec sortedLibraryIndices = sort_index(librarySizes, "descend");
  // arma::vec topLibraries = sortedLibraryIndices(span(0, N - 1));

  arma::vec repN(N);
  repN.fill(3);
  arma::vec LT1 = log2( (repN/librarySizes(span(0,N-1))) * 1000000 + 1);
  repN.fill(8);
  arma::vec LT2 = log2( (repN/librarySizes(span(0,N-1))) * 1000000 + 1);

  arma::mat dropoutCandidates(size(nData));
  arma::vec kernelEstimations(tagSize, fill::zeros);
  for (int i = 0; i < N; i++) {
    // need to call KDE
    // might not be able to do nice stuff, need to look at that
    // kde implemented is probably an approximation
    // Might need to find another library?
    // maybe I can use a different density estimation
    std::vector<double> kdeSamples =  conv_to<std::vector<double>>::from(nData.col(sortedLibraryIndices(i)));
    double mKDE = 0;
    double mIndex = 0;
    for (int j = 0; j < 512; j++)
    {
      // add args in kde to limit
      double kde = kerneldensity(kdeSamples.data(), LT2[i] + (j*LT2[i]/512), 1024);
      if (kde > mKDE)
      {
        mKDE = kde;
        mIndex = LT2[i] + (j*LT2[i]/512);
      }
    }

    double iterate = (mIndex - LT1(i))/512;
    mKDE = arma::datum::inf;
    for (int j = 0; j < 512; j++)
    {
      // add args in kde to limit
      double kde = kerneldensity(kdeSamples.data(), LT1(i) + (j*iterate), 1024);
      if (kde < mKDE)
      {
        mKDE = kde;
        mIndex = LT1(i) + (j*iterate);
      }
    }

    dTs(i) = mIndex;

    // kde_kernel.Evaluate()
    //
    //
    // for (int j = 0; )
  }

  // size sampleSize
  arma::vec dThreshold(sampleSize);
  dThreshold.fill(median(dTs));

  arma::mat tmpNData = nData.t();
  arma::mat tmpDropouts(nData.n_cols, nData.n_rows);
  // going through each column
  // don't need to go through each element, just go through each column
  int dIndex = 0;
  tmpNData.each_col([&](vec& datCol) {
    arma::uvec colCandidates = datCol < dThreshold;
    // tmpDropouts .insert_cols(colCandidates);
    tmpDropouts.col(dIndex) = conv_to<mat>::from(colCandidates);
    dIndex++;
  });
  dropoutCandidates = tmpDropouts.t();
  // for (int i = 0; i < tagSize; i++) {
  //   dropoutCandidates.insert_rows(i, (tmpNData(i) < dThreshold).t());
  // }

  // this might be unnecessary because of the column stuff
  // dropoutCandidates = dropoutCandidates.t();

  /*
    Imputation Weighting Threshold
  */

  //cutoff = 0.5
  arma::vec deleteDP(sampleSize);
  rowvec dpSums = arma::sum(dropoutCandidates, 0);
  int dpIndex = 0;
  for (int i = 0; i < tagSize; i++)
  {
    if (dpSums(i) == sampleSize) {
      deleteDP(i) = dpIndex;
      dpIndex++;
    }
  }

  arma::mat wThreshNData = nData;
  arma::mat wThreshDropoutCandidates = dropoutCandidates;

  if (dpIndex > 0)
  {
    // go through each row and remove the dp candidates
    for (int i = 0; i < dpIndex; i++)
    {
      // need to delete rows
      wThreshNData.shed_row(deleteDP(i));
      wThreshDropoutCandidates.shed_row(deleteDP(i));
    }
  }

  arma::vec dropoutRates = sum(dropoutCandidates)/sampleSize;
  arma::vec averLcpm(tagSize);
  for (int i = 0; i < tagSize; i++)
  {
    // going by column
    int cumSum = 0;
    int sumNum = 0;
    for (int j = 0; j < sampleSize; j++)
    {
      // means it is NOT a dropout candidate
      if (wThreshDropoutCandidates(i,j) == 0)
      {
        cumSum += wThreshNData(i,j);
        sumNum++;
      }
    }
    averLcpm(i) = cumSum/sumNum;
  }

  // use non linear least squares algorithm to find values of a and b (fits negative logistic function to dropouts vs expression); calls to coef just getting what we already solved for in the previous line
  double threshold = 13.70391; // hard-coding in threshold for now

  /*
    CIDR Dissimilarity Matrix
  */

  // // num of threads 4
  arma::mat Dist(sampleSize, sampleSize, fill::zeros);
  scPCA::cpp_dist(Dist, dropoutCandidates, nData, sampleSize, threshold);

  Dist = sqrt(Dist);
  Dist = Dist + Dist.t();

  /*
    Single-cell Principal Coordinates analysis
  */
  int distNRows = Dist.n_rows;
  double epsilon = 2.220446e-16;
  // might not need row namespace

  arma::mat diagMatrix(distNRows, distNRows, fill::eye);
  arma::mat oneMatrix(distNRows, distNRows, fill::ones);
  arma::mat ecMatrix = diagMatrix - (oneMatrix/distNRows);
  arma::mat editDist = -0.5 * (square(Dist));
  arma::mat center = ecMatrix * editDist * ecMatrix;
  arma::vec eigval;
  arma::mat eigvec;
  eig_sym(eigval, eigvec, center);

  eigval.for_each([&](mat::elem_type& val) {
    if (std::abs(val) < epsilon)
    {
      val = 0;
    }
  });

  // Flip to descending order
  eigval = reverse(eigval);
  eigvec = reverse(eigvec);

  // might need to cast to int
  int posK = sum(eigval > epsilon);
  arma::mat vectors(posK, posK, fill::zeros);
  int variationSize = 0;
  for (int i = 0; i < posK; i++)
  {
    vectors.col(i) = eigval(i) * eigvec.col(i);
    if (eigval(i) > 0)
    {
      variationSize++;
    }
  }
  arma::vec variation = eigval.head(variationSize);
  variation = variation/sum(variation);
  arma::mat PC = eigvec.cols(0, variationSize);

  /*
    Determine the optimal number of principal coordinates (nPC) for clustering (should put in separate function)
  */
  int l = int(0.9 * variationSize);
  if (l == 0)
  {
    l = variationSize;
  }

  int nPC;
  int npcDefault = 4;
  nPC = npcDefault;
  int cutoffDivisor = 10;
  arma::vec d = variation.head(variationSize - 1) - variation.tail(variationSize - 1);
  arma::vec descendD = sort(d, "descend");
  double max_d = d(0);
  double spread = mean(d)/mean(descendD.head(3)) * 100;
  if (spread > 15)
  {
    nPC = npcDefault;
  } else {
    if (spread > 10) {
      cutoffDivisor = 5;
    }
    double cutoff = max_d/cutoffDivisor;
    arma::vec groupSizes(variationSize, fill::zeros);
    int groupIndex = 0;
    for (int i = 1; i < variationSize; i++)
    {
      if (d(i-1) < cutoff)
      {
        groupSizes(groupIndex) += 1;
        if (groupSizes(groupIndex) > 7 || (i > 9 && groupSizes(groupIndex) > 3)) {
          if (groupIndex == 0)
          {
            //need return statement
            nPC = npcDefault;
          }
          nPC = groupSizes(groupIndex-1);

          if (nPC = 1)
          {
            nPC = npcDefault;
          }
          break;
        }

      }
      else {
        groupIndex += 1;
        groupSizes(groupIndex) += i;
      }
    }
  }

  /*
    Single-cell clustering
  */

  arma::mat expClustering = PC.cols(0, nPC-1);
  double n = 3*nPC + 3;




  return 0;
}
