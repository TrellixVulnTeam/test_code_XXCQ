#include <iostream>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <cmath>
#include <chrono>
#include <random>

using namespace std;

//! 顶点/节点//超点。顶点记录的是属于同一个block的未知数变量。通常一个block指的是，它包含的几个未知数是在一起被估计的，例如一个二次方程中的几个参数，
//! 或者一个camera
//! pose或者一个三维空间点。实现时，它必须继承自BaseVertex表示最普通的顶点类型。BaseVertex中定义了4个子类必须要定义的纯虚函数。
/*! BaseVertex模板参数：
    - 第1个参数是每个节点中待优化变量的维度（本问题中要优化3个变量）；
    - 第2个是盛放这些变量的数据类型。
    注意：g2o中已经定义好了很多常见的Vertex类型可以直接使用，例如BA问题中的三维空间点VertexSBAPointXYZ，以及相机位姿VertexSE3Expmap（以李代数的形式定义）等。
*/
class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    //! 类中有Eigen::Matrix变量并且将来会动态创建该类的对象时，加上下面这句Eigen提供的Macro
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /* 接下来是4个来自父类g2o::BaseVertex的**必须**要重载的函数。*/
    //! 来自父类的纯虚函数1：重置一个节点中全部参数为0。每个节点类继承了父类的_estimate变量，它的类型就是上面你自己定义的模板类型（之类就是Eigen::Vector3d)。
    //! 该函数通常只用于节点初始化时候调用。这里就是把Vector3d的3个元素都设成0。不过似乎没必要定义，因为Eigen::Vector3d在声明后会默认赋值为0。
    //! 注意：这里依然将该函数设置为虚函数，以保证后面再次继承该类的类也要定义该函数。其余的虚函数同理。
    virtual void setToOriginImpl() { _estimate << 0, 0, 0; }

    //! 更新节点。它实现的其实就是，已知当前的待更新未知数的值x_k和一个delta_x值，如何更新下一个x_{k+1}的值。本问题就比较简单，直接
    //! 就是x_{k+1} = x_k + delta_x。而对于一些复杂问题，例如BA中更新pose时，就不能是这么简单了。
    virtual void oplusImpl(const double* update)  // 更新
    {
        _estimate += Eigen::Vector3d(update);
    }

    // 从数据流中读取数据到节点。不过本问题中并不需要，可以留空，但是一定要定义。这里就写一个warning。
    virtual bool read(istream& in)
    {
        cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
        return false;
    }

    // 将数据输出到数据流。同样的，本问题中并不需要，可以留空，但是一定要定义。这里就写一个warning。
    virtual bool write(ostream& out) const
    {
        cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
        return false;
    }
};

//! 边/超边。继承的父类是误差模型，这里用的BaseUnaryEdge表示最普通的一元的边，它连接了两个顶点。
/*! BaseUnaryEdge中定义了3个子类必须要定义的纯虚函数。参见下面代码。
    BaseUnaryEdge的模板参数：
    - 第1个是误差项的维度。本问题是curve fitting，误差项就是一个scalar，因此就是1。
    - 第2个是存储测量值的变量类型。本题中的测量值自然就是二维点(x,y)，因此通常可以用Vector2d存储这个二维点，这样的话，
      该参数就应当是Vector2d。不过，这种做法的缺点是有点不够直观，因为我们要求的误差是y-f(x)。因此，这里还用另一种方法，
      就是给CurveFittingEdge额外定义一个成员变量_x来记录x。这样的话，测量值就只是y就行了，即此时这第2个参数就是double了。
      这两种方法都可以，取决于误差项的样子。
    - 第3个是边连接的顶点类型。自然就是上面刚刚定义的顶点类型了。
*/
class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}

    // 纯虚函数1：计算误差。
    void computeError()
    {
        // 首先获取节点中存储的未知数变量
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex*>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();  // 用estimate()来获取节点中存储的待优化变量的当前值
        // 计算误差。误差是存储在一个Eigen::Matrix变量_error中的，它的维度是根据BaseUnaryEdge模板的第1个参数定的。这里的维度就是1，因此就是_error(0,0)。
        // 这里的error()就是一个封装了_error的get()函数而已。
        // 另一个变量_measurement存储的是测量值。按照这里的代码，它就只是一个double，存储的就是y的值（就是你之后调用edge->setMeasurement()时的输入）。如果
        // 你上面使用的是Vector2d存储了(x,y)的值，那么这里的_measurement就是Vector2d（当然下面代码是要修改的）。类似的，measurement()也是封装了的get函数。
        error()(0, 0) = measurement() - std::exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
    }

    // 纯虚函数2和3：还是读取和输出函数，和上面的顶点中的定义类似。
    virtual bool read(istream& in)
    {
        cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
        return false;
    }
    virtual bool write(ostream& out) const
    {
        cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
        return false;
    }

public:
    double _x;  // 这只是x值，而y值是在 _measurement中，如果用Vector2d存储测量值，就不需要这个_x了。
};

int main(int argc, char** argv)
{
    double a = 1.0, b = 2.0, c = 1.0;           // 真实参数值
    int N = (argc == 1) ? 100 : atoi(argv[1]);  // 数据点个数
    double w_sigma = 2.0;                       // 噪声的标准差，即sigma

    // 设置高斯噪声生成器。这里使用STL中的normal_distribution类
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);  // 使用时间作为seed，否则生成的随机数在每次程序运行都是一批相同的数
    double kNoiseSigma = 1.0;                               // 噪声的标准差Standard Deviation
    std::normal_distribution<double> dist(0, kNoiseSigma);  // 第1个参数是mean，第2个是标准差
    // cv::RNG rng;               // OpenCV的随机数

    vector<double> x_data, y_data;  // 数据
    for (int i = 0; i < N; i++)
    {
        double x = double(i) / N;
        x_data.push_back(x);
        // 增加高斯噪声到测量值。被注释的是使用OpenCV提供的高斯函数。
        y_data.push_back(exp(a * x * x + b * x + c) + dist(generator));
        // y_data.push_back(exp(a * x * x + b * x + c) + rng.gaussian(w_sigma));
    }

    /* 建立graph开始优化 */
    // 下面注释的是SLAM十四讲原书中的源代码，但编译会出错。有可能是代码比较老，而g2o后来又做了修改，将模板参数从固定的维度类型改成了为Eigen::Dynamic类型，
    // 好处是我们不用再care维度的设置了。
    /*
       typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> Block;  // 每个误差项优化变量维度为3，误差值维度为1
       Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();  // 线性方程求解器
       Block* solver_ptr = new Block(linearSolver);                                                  // 矩阵块求解器
    */
    // 一般 general 问题中 MyBlockSolver 是固定成下面这句的样子。当然 g2o 已经有一些 typedef 定义了常见的几个类型供类似
    // BA 问题使用
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<Eigen::Dynamic, Eigen::Dynamic>> MyBlockSolver;
    // 这里使用Dense Solver，同理还有稀疏阵类型LinearSolverSparse，通常用于BA等问题的优化
    typedef g2o::LinearSolverDense<MyBlockSolver::PoseMatrixType> MyLinearSolver;  // 嵌套了上面的BlockSolver
    // 设计优化类型，这里使用LM方法。还也可以使用高斯牛顿法：OptimizationAlgorithmGaussNewton，或者Dogleg法：OptimizationAlgorithmDogleg等。
    // 注意，最终的solver是要按照顺序嵌套两层solver的，最底层是linear solver，接着上面是block solver，最后是最终solver。
    std::unique_ptr<MyLinearSolver> linear_solver = g2o::make_unique<MyLinearSolver>();
    std::unique_ptr<MyBlockSolver> block_solver = g2o::make_unique<MyBlockSolver>(std::move(linear_solver));
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));
    // 其实上面三行代码用下面一行就够了，不过上面分开写更加容易看懂。
    // g2o::OptimizationAlgorithmLevenberg* solver =
    //     new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<MyBlockSolver>(g2o::make_unique<MyLinearSolver>()));

    g2o::SparseOptimizer optimizer;  // 图模型
    optimizer.setAlgorithm(solver);  // 设置求解器
    optimizer.setVerbose(true);      // 打开调试输出

    // 往图中增加顶点
    CurveFittingVertex* v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(0, 0, 0));  // 待优化变量的初始值，它不能离全局最优解太远，否则会陷入局部最优
    v->setId(0);                               // 本问题中我们就1个顶点
    optimizer.addVertex(v);

    // 往图中增加边
    for (int i = 0; i < N; i++)
    {
        CurveFittingEdge* edge = new CurveFittingEdge(x_data[i]);  // 定义边的时候把x作为成员变量了
        edge->setId(i);                                            // 每条边有一个id
        edge->setVertex(0, v);  // 本问题中只有一个顶点，因此每条边都是和自己相连。故，这里无需设置setVertex(1,...)
        edge->setMeasurement(y_data[i]);  // 观测值，这里就只有y值了
        // 每一条边都有一个信息矩阵（即e^T\Omega
        // e正中间的\Omega，即协方差矩阵的逆矩阵)，用于平衡不同的测量值变量之间的重要性的。
        // 通常它都是单位阵，即全部测量值都一样。
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
        optimizer.addEdge(edge);
    }

    // 执行优化
    cout << "start optimization" << endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(100);  // 输入是迭代次数
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    // 输出优化值
    Eigen::Vector3d abc_estimate = v->estimate();
    cout << "estimated model: " << abc_estimate.transpose() << endl;

    return 0;
}
