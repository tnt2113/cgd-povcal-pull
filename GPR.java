public void predict(int predictFuture) throws IllegalArgumentException, IllegalAccessException, InvocationTargetException {

    RealMatrix y = MatrixUtils.createColumnRealMatrix(this.targets);

    double mean = StatUtils.mean(this.targets);

    for(int i = 0; i < this.targets.length;i++)
    {
        targets[i] -= mean;
    }


    RealMatrix K =  MatrixUtils.createRealMatrix(cov);


    //identity matrix for I
    RealMatrix k_eye = MatrixUtils.createRealIdentityMatrix(cov.length);


    //choleski(K + sigman^2*I)
    CholeskyDecomposition L = null;
    try {
        L = new CholeskyDecompositionImpl(
                K.add(
                        k_eye.scalarMultiply(Math.pow(parameter[2], 2))
                        )
                );
    } catch (NonSquareMatrixException e) {
        e.printStackTrace();
    } catch (NotSymmetricMatrixException e) {
        e.printStackTrace();
    } catch (NotPositiveDefiniteMatrixException e) {
        e.printStackTrace();
    }

    //inverse of Ltranspose for left devision
    RealMatrix L_transpose_1 = new LUDecompositionImpl(L.getLT()).getSolver().getInverse();
    //inverse of Ltranspose for left devision
    RealMatrix L_1 = new LUDecompositionImpl(L.getL()).getSolver().getInverse();



    //alpha = L'\(L\y)
    RealMatrix alpha = L_transpose_1.multiply(L_1).multiply(y);


    double L_diag = 0.0;

    for(int i = 0; i < L.getL().getColumnDimension();i++)
    {
        L_diag += Math.log(L.getL().getEntry(i, i)); 
    }

    double logpyX = - y.transpose().multiply(alpha).scalarMultiply(0.5).getData()[0][0]
                    - L_diag
                    - predictFuture * Math.log(2 * Math.PI) * 0.5;


    double[] fstar = new double[targets.length + predictFuture];
    double[] V = new double[targets.length + predictFuture];

    for(int i = 0;i < targets.length + predictFuture;i++)
    {

        double[] kstar = new double[targets.length];

        for(int j = 0; j < targets.length;j++)
        {
            double covar = (Double)covMethod.invoke(this,j,i);
            kstar[j] = covar;
        }

        //f*=k_*^T * alpha
        fstar[i] = MatrixUtils.createColumnRealMatrix(kstar).transpose().multiply(alpha).getData()[0][0];
        fstar[i] += mean;
        //v = L\k_*
        RealMatrix v = L_1.multiply(MatrixUtils.createColumnRealMatrix(kstar));

        //V[fstar] = k(x_*,x_*) - v^T*v
        double covar = (Double)covMethod.invoke(this,i,i);

        V[i] = covar - v.transpose().multiply(v).getData()[0][0] + Math.pow(parameter[2],2);
    }

    this.predicted_mean = fstar;
    this.predicted_variance = V;
}

