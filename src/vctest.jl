
using Random
using RCall
function vctest(y, X, V;
                    bInit::Array{Float64, 1} = Float64[],
                    devices::String = "CPU",
                    nMMmax::Int = 0,
                    nBlockAscent::Int = 1000,
                    nNullSimPts::Int = 10000,
                    nNullSimNewtonIter::Int = 15,
                    tests::String = "eLRT",
                    tolX::Float64 = 1e-6,
                    vcInit::Array{Float64, 1} = Float64[],
                    Vform::String = "whole",
                    pvalueComputings::String = "chi2",
                    WPreSim::Array{Float64, 2} = [Float64[] Float64[]],
                    PrePartialSumW::Array{Float64, 2} = [Float64[] Float64[]],
                    PreTotalSumW::Array{Float64, 2} = [Float64[] Float64[]],
                    partialSumWConst::Array{Float64, 1} = Float64[],
                    totalSumWConst::Array{Float64, 1} = Float64[],
                    windowSize::Int = 50,
                    partialSumW::Array{Float64, 1} = Float64[],
                    totalSumW::Array{Float64, 1} = Float64[],
                    lambda::Array{Float64, 2} = [Float64[] Float64[]],
                    W::Array{Float64, 2} = [Float64[] Float64[]],
                    nPreRank::Int = 20,
                    tmpmat0::Array{Float64, 2} = [Float64[] Float64[]],
                    tmpmat1::Array{Float64, 2} = [Float64[] Float64[]],
                    tmpmat2::Array{Float64, 2} = [Float64[] Float64[]],
                    tmpmat3::Array{Float64, 2} = [Float64[] Float64[]],
                    tmpmat4::Array{Float64, 2} = [Float64[] Float64[]],
                    tmpmat5::Array{Float64, 2} = [Float64[] Float64[]],
                    denomvec::Array{Float64, 1} = Float64[],
                    d1f::Array{Float64, 1} = Float64[],
                    d2f::Array{Float64, 1} = Float64[],
                    offset::Int = 0,
                    nPtsChi2::Int = 300,
                    simnull::Array{Float64, 1} = Float64[])

  # VCTEST Fit and test for the nontrivial variance component
  #
  # [SIMNULL] = VCTEST(y,X,V) fits and then tests for $H_0:sigma_1^2=0$ in
  # the variance components model Y ~ N(X \beta, vc0*I + vc1*V).
  #
  #   INPUT:
  #       y - response vector
  #       X - design matrix for fixed effects
  #       V - the variance component being tested
  #
  #   Optional input name-value pairs:
  #       'nBlockAscent' - max block ascent iterations, default is 1000
  #       'nMMmax' - max MM iterations, default is 10 (eLRT) or 1000 (eRLRT)
  #       'nSimPts'- # simulation samples, default is 10000
  #       'test'- 'eLRT'|'eRLRT'|'eScore'|'none', the requested test, it also
  #           dictates the estimation method
  #       'tolX' - tolerance in change of parameters, default is 1e-5
  #       'Vform' - format of intpu V. 'whole': V, 'half': V*V', 'eigen':
  #           V.U*diag(V.E)*V.U'. Default is 'whole'. For computational
  #           efficiency,  a low rank 'half' should be used whenever possible
  #
  #
  #   Output:
  #       b - estimated regression coeffiicents in the mean component
  #       vc0 - estiamted variance component for I
  #       vc1 - estiamted variance component for V
  #       stats - other statistics

  # check dimensionalities
  n = length(y);
  if size(V, 1) != n
    error("vctest:wrongdimV\n", "dimension of V does not match that of X");
  end

  # set up default matrices
  Random.seed!(0)
  if isempty(WPreSim)
      WPreSim = randn(windowSize, nNullSimPts);
      for idxWc = 1 : nNullSimPts
          for idxWr = 1 : windowSize
          global WPreSim[idxWr, idxWc] = WPreSim[idxWr, idxWc] * WPreSim[idxWr, idxWc];
          end
      end
  end

  if isempty(partialSumW)
      partialSumW = Array{Float64}(undef, nNullSimPts);
  end
  if isempty(totalSumW)
      totalSumW= Array{Float64}(undef, nNullSimPts);
  end
  if isempty(lambda)
      lambda = Array{Float64}(undef, 1, nNullSimPts);
  end
  if isempty(lambda)
      W = Array{Float64}(undef, windowSize, nNullSimPts);
  end
  if isempty(tmpmat0)
      tmpmat0 =Array{Float64}(undef, windowSize, nNullSimPts);
  end
  if isempty(tmpmat1)
      tmpmat1 = Array{Float64}(undef, windowSize, nNullSimPts);
  end
  if isempty(tmpmat2)
      tmpmat2 = Array{Float64}(undef,windowSize, nNullSimPts);
  end
  if isempty(tmpmat3)
      tmpmat3 = Array{Float64}(undef,windowSize, nNullSimPts);
  end
  if isempty(tmpmat4)
      tmpmat4 = Array{Float64}(undef,windowSize, nNullSimPts);
  end
  if isempty(tmpmat5)
      tmpmat5 = Array{Float64}(undef, windowSize, nNullSimPts);
  end
  if isempty(denomvec)
      denomvec = Array{Float64}(undef,nNullSimPts);
  end
  if isempty(d1f)
      d1f = Array{Float64}(undef,nNullSimPts);
  end
  if isempty(d2f)
      d2f = Array{Float64}(undef,nNullSimPts);
  end
  if isempty(W)
      W = Array{Float64}(undef, windowSize, nNullSimPts);
  end
  if isempty(simnull)
      simnull = Array{Float64}(undef, nNullSimPts);
  end

  # set default maximum MM iteration
  if nMMmax == 0 && tests == "eLRT"
    nMMmax = 10;
  elseif nMMmax == 0 && tests == "eRLRT"
    nMMmax = 1000;
  end

  # SVD of X
  if isempty(X)
    rankX = 0;
    X = reshape(X, n, 0);
    X = convert(Array{Float64, 2}, X);
    # LRT is same as RLRT if X is null
    if tests == "eLRT"
      tests = "eRLRT";
    end
  else
    if tests == "eRLRT"
      (UX, svalX) = svd(X; full = true);  ## jz: svd(X, thin = false) -> svd(X; full = true)
    else
      (UX, svalX) = svd(X);
    end
    rankX = sum(svalX .> n * eps(svalX[1]));  ## jz: countnz -> sum
  end

  # eigendecomposition of V
  if Vform == "whole"
    (evalV, UV) = eigen(V);   ## jz: eig(V)->eigen(V)
    rankV = sum(evalV .> n * eps(sqrt(maximum(evalV)))); ## jz: countnz -> sum
    sortIdx = sortperm(evalV, rev = true);
    evalV = evalV[sortIdx[1:rankV]];
    UV = UV[:, sortIdx[1:rankV]];
    if tests == "eLRT" || tests == "eRLRT"
      wholeV = V;
    end
  elseif Vform == "half"
    (UVfull, tmpevalVfull) = svd(V; full = true);  ## jz: svd(X, thin = false) -> svd(X; full = true)
    evalVfull = zeros(n);
    pevalVfull = pointer(evalVfull);
    ptmpevalV = pointer(tmpevalVfull);
    BLAS.blascopy!(length(tmpevalVfull), ptmpevalV, 1, pevalVfull, 1);
    rankV = sum(evalVfull .> n * eps(maximum(evalVfull)));
    # evalVfull = evalVfull .^ 2; ##jz: comment
    evalV = evalVfull[1:rankV];
    UV = UVfull[:, 1:rankV];
    if tests == "eLRT" || tests == "eRLRT"
      wholeV = *(V, V');
    end
  elseif Vform == "eigen"
    (evalV, UV) = V;
    # evalV = V.eval;  ##jz: comment
    rankV = sum(evalV .> n * eps(sqrt(maximum(evalV))));
    sortIdx = sortperm(evalV, rev = true);
    evalV = evalV[sortIdx[1:rankV]];
    UV = UV[:, sortIdx[1:rankV]];
    if tests == "eLRT" || tests == "eRLRT"
      wholeV = *(UV * Diagonal(evalV), UV');
    end
  end

  # obtain eigenvalues of (I-PX)V(I-PX)
  if !isempty(X) || tests == "eScore"
    #sqrtV = UV .* sqrt(evalV)';
    sqrtV = similar(UV);
    psqrtV = pointer(sqrtV);
    pUV = pointer(UV);
    BLAS.blascopy!(n*rankV, pUV, 1, psqrtV, 1);
    sqrtV = sqrtV * Diagonal(sqrt.(evalV));
    # scale!(sqrtV, sqrt(evalV));
  end
  if isempty(X)
    evalAdjV = evalV;
  else
    subUX = Array{Float64}(undef, n, rankX);   ##jz: Array{Float64}(n, rankX); -> Array{Float64}(undef, n, rankX)
    psubUX = pointer(subUX);
    pUX = pointer(UX);
    BLAS.blascopy!(n*rankX, pUX, 1, psubUX, 1);
    mat1 = BLAS.gemm('T', 'N', 1.0, subUX, sqrtV);
    mat2 = BLAS.gemm('N', 'N', 1.0, subUX, mat1);
    (UAdjV, evalAdjV) = svd(sqrtV - mat2; full = true);
    if isempty(evalAdjV)
      evalAdjV = Float64[];
    else
      evalAdjV = evalAdjV[evalAdjV .> n * eps(maximum(evalAdjV))] .^ 2;
    end
  end
  rankAdjV = length(evalAdjV);

  # fit the variance component model
  if tests == "eLRT"

    # estimates under null model
    bNull = X \ y;
    rNull = y - X * bNull;
    vc0Null = norm(rNull) ^ 2 / n;
    Xrot = UVfull' * X;
    yrot = UVfull' * y;
    loglConst = - 0.5 * n * log(2.0 * pi);

    # set starting point
    if isempty(bInit)
      b = copy(bNull);
      r = copy(rNull);
    else
      b = copy(bInit);
      r = y - X * b;
    end
    if isempty(vcInit)
      vc0 = norm(r) ^ 2 / n;
      vc1 = 1;
      wt = 1.0 ./ sqrt.(vc1 * evalVfull .+ vc0);  ##jz: add .
    else
      vc0 = vcInit[1];
      # avoid sticky boundary
      if vc0 == 0
        vc0 = 1e-4;
      end
      vc1 = vcInit[2];
      if vc1 == 0
        vc1 = 1e-4;
      end
      wt = 1.0 ./ sqrt.(vc1 * evalVfull .+ vc0);
      Xnew = Diagonal(wt) * Xrot;
      ynew = wt .* yrot;
      b = Xnew \ ynew;
    end

    # update residuals according supplied var. components
    r = y - LinearAlgebra.BLAS.gemv('N', X, b)
    rnew =  LinearAlgebra.BLAS.gemv('T', UVfull, r);
    logl = loglConst + sum(log, wt) - 0.5 * sum(abs2,rnew .* wt);

    nIters = 0;

    denvec = Float64[];
    # global vc0, vc1, denvec, numvec, logl, loglOld, r, rnew, wt, b, Xnew, ynew
    for iBlockAscent = 1:nBlockAscent

      nIters = iBlockAscent;
      # update variance components
      for iMM = 1:nMMmax
         denvec = 1.0 ./ (vc1 * evalVfull .+ vc0);
         numvec = rnew .* denvec;
         vc0 = vc0 * sqrt.(sum(abs2,numvec) / sum(abs,denvec));
         vc1 = vc1 * sqrt.( dot(numvec, numvec .* evalVfull) /
                           sum(abs, evalVfull .* denvec) );
        #wt = 1.0 ./ sqrt(vc1 * evalVfull + vc0);
      end
      wt = sqrt.(denvec);
      Xnew = Diagonal(wt) * Xrot;
      ynew = wt .* yrot;
      b = Xnew \ ynew;
      r = y - LinearAlgebra.BLAS.gemv('N', X, b)
      rnew =  LinearAlgebra.BLAS.gemv('T', UVfull, r);

      # stopping criterion
      loglOld = logl;
      logl = loglConst + sum(log, wt) - 0.5 * LinearAlgebra.BLAS.dot(length(evalVfull), rnew .^ 2, 1, denvec, 1);
      if abs(logl - loglOld) < tolX * (abs(logl) + 1.0)
            break
      end

    end


    # log-likelihood at alternative
    logLikeAlt = logl;
    # log-likelihood at null
    logLikeNull = loglConst - 0.5 * n * log(vc0Null) -  0.5 * sum(rNull .^ 2) / vc0Null;

    if logLikeNull >= logLikeAlt
      vc0 = vc0Null;
      vc1 = 0;
      logLikeAlt = logLikeNull;
      b = bNull;
      r = rNull;
    end

    # LRT test statistic
    statLRT = 2 * (logLikeAlt - logLikeNull);

    # obtain p-value for testing vc1=0
    vc1_pvalue = vctestnullsim(statLRT, evalV, evalAdjV, n, rankX,
                               WPreSim, device = devices,
                               nSimPts = nNullSimPts,
                               nNewtonIter = nNullSimNewtonIter,
                               test = "eLRT",
                               pvalueComputing = pvalueComputings,
                               PrePartialSumW = PrePartialSumW,
                               PreTotalSumW = PreTotalSumW,
                               partialSumWConst = partialSumWConst,
                               totalSumWConst = totalSumWConst,
                               windowSize = windowSize,
                               partialSumW = partialSumW,
                               totalSumW = totalSumW,
                               lambda = lambda, W = W,
                               nPreRank = nPreRank,
                               tmpmat0 = tmpmat0, tmpmat1 = tmpmat1,
                               tmpmat2 = tmpmat2, tmpmat3 = tmpmat3,
                               tmpmat4 = tmpmat4, tmpmat5 = tmpmat5,
                               denomvec = denomvec,
                               d1f = d1f, d2f = d2f, offset = offset,
                               nPtsChi2 = nPtsChi2, simnull = simnull);

    # return values
    return b, vc0, vc1, vc1_pvalue;

  elseif tests == "eRLRT"

    if isempty(X)
      ytilde = y;
      rankBVB = rankV;
      evalBVB = evalV;
      UBVB = UV;
    else
      # obtain a basis of N(X')
      B = UX[:, rankX+1:end];
      ytilde = B' * y;

      # eigen-decomposition of B'VB and transform data
      (UBVB, evalBVB) = svd(B' * sqrtV);
      rankBVB = sum(evalBVB .> n * eps(maximum(evalBVB)));   ##jz: countnz ->sum
      evalBVB = evalBVB[1:rankBVB] .^ 2;
      UBVB = UBVB[:, 1:rankBVB];
    end
    resnew = UBVB' * ytilde;
    normYtilde2 = norm(ytilde) ^ 2;
    deltaRes = normYtilde2 - norm(resnew) ^ 2;

    # set initial point
    # TODO: is there better initial point?
    vc0Null = normYtilde2 / length(ytilde);
    if isempty(vcInit)
      vc0 = vc0Null;
      vc1 = 1.0;
    else
      vc0 = vcInit[1];
      vc1 = vcInit[2];
    end

    # MM loop for estimating variance components
    nIters = 0;
    #tmpvec = Array(Float64, rankBVB);
    #denvec = Array(Float64, rankBVB);
    #numvec = Array(Float64, rankBVB);
    tmpvec = Float64[];
    for iMM = 1:nMMmax   ##      global vc1, vc0
      nIters = iMM;
      #tmpvec = vc0 + vc1 * evalBVB;
      #tmpvec = evalBVB;
      #numvecSum = 0.0;
      #denvecSum = 0.0;
      #numvecProdSum = 0.0;
      #denvecProdSum = 0.0;
      tmpvec = BLAS.scal(rankBVB, vc1, evalBVB, 1);
      #for i = 1 : rankBVB
      #  tmpvec[i] += vc0;
      #  denvec[i] = 1 / tmpvec[i];
      #  numvec[i] = (resnew[i] * denvec[i]) ^ 2;
      #  numvecSum += numvec[i];
      #  denvecSum += denvec[i];
      #  numvecProdSum += evalBVB[i] * numvec[i];
      #  denvecProdSum += evalBVB[i] * denvec[i];
      #end
      tmpvec .+= vc0;    ## jz: add .
      denvec = 1 ./ tmpvec;
      numvec = (resnew .* denvec) .^ 2;
      vcOld = [vc0 vc1];
      vc0 = sqrt( (vc0 ^ 2 * sum(numvec) + deltaRes) /
                   (sum(denvec) + (n - rankX - rankBVB) / vc0) );
      #vc0 = sqrt( (vc0 ^ 2 * numvecSum + deltaRes) /
      #             (denvecSum + (n - rankX - rankBVB) / vc0) );
      vc1 = vc1 * sqrt( sum(evalBVB .* numvec) / sum(evalBVB .* denvec) );
      #vc1 = vc1 * sqrt( numvecProdSum / denvecProdSum );
      # stopping criterion
      if norm([vc0 vc1] - vcOld) <= tolX * (norm(vcOld) + 1)
        break;
      end
    end
    global vc0, vc1, tmpvec


    # restrictive log-likelihood at alternative

    loglConst = - 0.5 * (n - rankX) * log(2 * pi);
    logLikeAlt =  loglConst - 0.5 * sum(log.(tmpvec)) -
      0.5 * (n - rankX - rankBVB) * log(vc0) - 0.5 * normYtilde2 / vc0 +
      0.5 * sum(resnew .^ 2 .* (1 / vc0 .- 1 ./ (tmpvec)));
    # restrictive log-likelihood at null
    logLikeNull = - 0.5 * (n - rankX) * (log(2 * pi) + log(vc0Null)) -
      0.5 / vc0Null * normYtilde2;
    if logLikeNull >= logLikeAlt
      vc0 = vc0Null;
      vc1 = 0;
      logLikeAlt = logLikeNull;
    end

    # RLRT test statitic
    statRLRT = 2 * (logLikeAlt - logLikeNull);

    # obtain p-value for testing vc1=0
    vc1_pvalue = vctestnullsim(statRLRT, evalV, evalAdjV, n, rankX,
                               WPreSim, device = devices,
                               nSimPts = nNullSimPts,
                               nNewtonIter = nNullSimNewtonIter,
                               test = "eRLRT",
                               pvalueComputing = pvalueComputings,
                               PrePartialSumW = PrePartialSumW,
                               PreTotalSumW = PreTotalSumW,
                               partialSumWConst = partialSumWConst,
                               totalSumWConst = totalSumWConst,
                               windowSize = windowSize,
                               partialSumW = partialSumW,
                               totalSumW = totalSumW,
                               lambda = lambda, W = W,
                               nPreRank = nPreRank,
                               tmpmat0 = tmpmat0, tmpmat1 = tmpmat1,
                               tmpmat2 = tmpmat2, tmpmat3 = tmpmat3,
                               tmpmat4 = tmpmat4, tmpmat5 = tmpmat5,
                               denomvec = denomvec,
                               d1f = d1f, d2f = d2f, offset = offset,
                               nPtsChi2 = nPtsChi2, simnull = simnull);

    # estimate fixed effects
    if isempty(X)
      b = zeros(0);
    else
      Xrot = UV' * X;
      yrot = UV' * y;
      wt = 1.0 ./ sqrt.(vc1 * evalV .+ vc0);
      Xnew = Diagonal(wt) * Xrot;
      ynew = wt .* yrot;
      b = Xnew \ ynew;
    end

    # return values
    return b, vc0, vc1, vc1_pvalue;

  elseif tests == "eScore"

    # fit the null model
    b = X \ y;
    r = y - X * b;
    vc0 = norm(r) ^ 2 / n;
    vc1 = 0;

    # score test statistic
    #statScore = norm(r' * sqrtV) ^ 2 / norm(r) ^ 2;
    statScore = norm(LinearAlgebra.BLAS.gemv('T', sqrtV, r)) ^ 2 / norm(r) ^ 2;
    #statScore = max(statScore, sum(evalV) / n);
    #=
    # obtain p-value for testing vc1=0
    vc1_pvalue = vctestnullsim(statScore, evalV, evalAdjV, n, rankX,
                               WPreSim, test = "eScore",
                               nSimPts = nNullSimPts,
                               pvalueComputing = pvalueComputings,
                               nNewtonIter = nNullSimNewtonIter,
                               device = devices,
                               PrePartialSumW = PrePartialSumW,
                               PreTotalSumW = PreTotalSumW,
                               partialSumWConst = partialSumWConst,
                               totalSumWConst = totalSumWConst,
                               windowSize = windowSize,
                               partialSumW = partialSumW,
                               totalSumW = totalSumW,
                               lambda = lambda, W = W,
                               nPreRank = nPreRank,
                               tmpmat0 = tmpmat0, tmpmat1 = tmpmat1,
                               tmpmat2 = tmpmat2, tmpmat3 = tmpmat3,
                               tmpmat4 = tmpmat4, tmpmat5 = tmpmat5,
                               denomvec = denomvec,
                               d1f = d1f, d2f = d2f, offset = offset,
                               nPtsChi2 = nPtsChi2, simnull = simnull);
    =#
    if statScore <= sum(evalV) / n
      vc1_pvalue = 1.0;
    else
      w = [evalAdjV .- statScore; - statScore];
      dfvec = [ones(Int, rankAdjV); n - rankX - rankAdjV];
      # fun(x) = imag(prod((1 - 2 .* w * x * im) .^ (- dfvec / 2)) / x);
      # (vc1_pvalue, err) = quadgk(fun, 0, Inf);
      @rput w
      @rput dfvec
      R"
      require(pracma)
      f<-function(x){Im(prod((1-1i*(2*w*x))^(- dfvec/2))/x)}
      vc1_pvalue <- quadinf(f, 0, Inf)$Q
      "
      @rget vc1_pvalue
      vc1_pvalue = 0.5 + vc1_pvalue / pi;

    end

    # return values
    return b, vc0, vc1, vc1_pvalue;

  end

end
