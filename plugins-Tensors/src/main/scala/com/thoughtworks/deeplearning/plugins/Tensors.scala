package com.thoughtworks.deeplearning.plugins

import java.util.{Collections, IdentityHashMap}
import java.util.concurrent.Callable

import com.dongxiguo.fastring.Fastring.Implicits._
import com.google.common.cache._
import com.thoughtworks.compute.OpenCL
import com.thoughtworks.continuation._
import com.thoughtworks.expressions.api.{Arrays, Floats}
import com.thoughtworks.expressions.opencl.Context
import com.thoughtworks.expressions.opencl.Context.GlobalContext
import com.thoughtworks.expressions.tree.{FloatArrayTrees, StructuralTrees}
import com.thoughtworks.feature.Factory
import com.thoughtworks.future._
import com.thoughtworks.raii.asynchronous._
import com.thoughtworks.raii.covariant._
import org.apache.commons.math3.linear._

import scala.annotation.tailrec
import scala.concurrent.ExecutionContext
import scala.language.existentials
import scalaz.Tags.Parallel
import scalaz.std.list._
import scalaz.syntax.all._
import scalaz.syntax.tag._

// TODO: Rename to VirtualTensors, like virtual-dom
trait Tensors extends OpenCL {

  protected val trees: FloatArrayTrees with StructuralTrees { type Category = Floats with Arrays } =
    Factory[FloatArrayTrees with StructuralTrees].newInstance()
  import trees._

  private def upvalues(tree: TreeApi): List[Parameter] = {
    val traversed: java.util.Set[TreeApi] = Collections.newSetFromMap(new IdentityHashMap)
    val builder = List.newBuilder[Parameter]
    def buildParameterList(tree: TreeApi): Unit = {
      tree match {
        case tree: Parameter =>
          builder += tree
        case _ =>
          val productArity = tree.productArity
          @tailrec def loop(i: Int): Unit = {
            if (i < productArity) {
              tree.productElement(i) match {
                case child: TreeApi =>
                  val isNew = traversed.add(tree)
                  if (isNew) {
                    buildParameterList(child)
                  }
                case _ =>
              }
              loop(i + 1)
            }
          }
          loop(0)
      }

    }
    buildParameterList(tree)
    builder.result()
  }

  // The scalar data type is hard-coded Float at the moment. FIXME: Allow other types in the future
  trait PendingBuffer {
    def event: Event

    def buffer: DeviceBuffer[Float]
  }

  sealed trait Tensor { thisTensor =>

//    def debuggingInformation: Implicitly[DebuggingInformation]

    def shape: Seq[Int]

    def closure: ValueTerm

    def enqueue: Do[PendingBuffer]

  }

  trait CompiledKernel extends MonadicCloseable[UnitContinuation] {
    def run(parameters: List[Parameter]): Do[PendingBuffer]
  }

  protected def kernelCacheBuilder: CacheBuilder[ValueTerm, CompiledKernel] = {
    CacheBuilder
      .newBuilder()
      .removalListener(new RemovalListener[ValueTerm, CompiledKernel] {
        def onRemoval(notification: RemovalNotification[ValueTerm, CompiledKernel]): Unit = {
          val compiledKernel = notification.getValue
          compiledKernel.monadicClose.blockingAwait
        }
      })
  }

  protected val kernelCache: Cache[ValueTerm, CompiledKernel] = kernelCacheBuilder.build()

  protected implicit val executionContext: ExecutionContext

  private def clearCache: UnitContinuation[Unit] = UnitContinuation.execute {
    kernelCache.invalidateAll()
    kernelCache.cleanUp()
  }

  override def monadicClose: UnitContinuation[Unit] = {
    clearCache >> super.monadicClose
  }

  /** An intermediate expression of tensor that can be composed into a more complex expression.
    *
    * @note When this [[InlineTensor]] is referenced more than one expressions,
    *       the computation for the tensor may be evaluated more than once.
    * @see [[force]] to create a tensor that will cache the result.
    */
  trait InlineTensor extends Tensor {
    def force: BufferedTensor = {
      new {
//        val debuggingInformation: Implicitly[DebuggingInformation] = InlineTensor.this.debuggingInformation
        val shape: Seq[Int] = InlineTensor.this.shape
        val enqueue: Do[PendingBuffer] = InlineTensor.this.enqueue
      } with BufferedTensor
    }

    lazy val enqueue: Do[PendingBuffer] = {
      val compiledKernel = kernelCache.get(
        closure,
        new Callable[CompiledKernel] {
          def call(): CompiledKernel = {

            val alphConversionContext = new AlphaConversionContext
            val convertedTerm = closure.tree.alphaConversion(alphConversionContext).asInstanceOf[ValueTerm]

            val sourceCode = {
              val globalContext = new GlobalContext
              val functionContext = Factory[Context].newInstance(globalContext)

              val exportContext = new ExportContext
              val kernelBody = convertedTerm.tree.export(functionContext, exportContext)

              val kernelParameters = upvalues(closure.tree).map { upvalue: Parameter =>
                exportContext.get(alphConversionContext.get(upvalue)).asInstanceOf[functionContext.Term]
              }
              fastraw"""
              ${globalContext.globalDeclarations}
              ${globalContext.globalDefinitions}
              ${functionContext.generateKernelSourceCode("kernel", shape.length, kernelParameters, Seq(kernelBody))}
              """
            }

            val program = createProgramWithSource(sourceCode)
            program.build()

            val compiledKernel = new CompiledKernel {

              def monadicClose: UnitContinuation[Unit] = program.monadicClose

              def run(upvalues: List[Parameter]): Do[PendingBuffer] = {
                // TODO: Manage life cycle of upvalues more delicately
                // e.g. a buffer should be release as soon as possible if it is a dependency of another buffer,
                // e.g. however, it can be hold longer time if it is dependencies of many other buffers.

                upvalues
                  .traverse[ParallelDo, PendingBuffer] { tree =>
                    Parallel(tree.asInstanceOf[Parameter].id.asInstanceOf[Tensor].enqueue)
                  }
                  .unwrap
                  .intransitiveFlatMap {
                    arguments: List[PendingBuffer] =>
                      Do.monadicCloseable(program.createFirstKernel()).intransitiveFlatMap { kernel: Kernel =>
                        allocateBuffer[Float](shape.product).flatMap { outputBuffer =>
                          for ((arugment, i) <- arguments.view.zipWithIndex) {
                            kernel(i) = arugment.buffer
                          }
                          kernel(arguments.length) = outputBuffer
                          kernel.enqueue(shape.view.map(_.toLong): _*).map { event0 =>
                            new PendingBuffer {
                              val event: Event = event0
                              val buffer: DeviceBuffer[Float] = outputBuffer
                            }
                          }
                        }
                      }
                  }

              }
            }
            kernelCache.put(convertedTerm, compiledKernel)
            compiledKernel
          }
        }
      )

      compiledKernel.run(upvalues(closure.tree)).shared
    }
  }

  trait TransformedTensor extends InlineTensor {

    def checkpoint: Tensor

    /** A matrix that describes the transformation of coordinate.
      *
      * The matrix size is __number of dimensions of original tensor × number of dimensions of new tensor__.
      */
    def matrix: RealMatrix

    val closure: ValueTerm = {
      array.parameter(checkpoint, float, shape: _*).transform(matrix).extract
    }
  }

  trait BufferedTensor extends Tensor {
    val closure: ValueTerm = {
      array.parameter(this, float, shape: _*).extract
    }
  }

  def translate(previousTensor: Tensor, offset: Seq[Double]): Tensor = {
    translate(previousTensor, offset, previousTensor.shape)
  }

  def translate(previousTensor: Tensor,
                offset: Seq[Double],
                newShape: Seq[Int]) /*(implicit debuggingInformation0: Implicitly[DebuggingInformation])*/: Tensor = {
    if (offset.length != previousTensor.shape.length) {
      throw new IllegalArgumentException
    }

    previousTensor match {
      case previousTensor: TransformedTensor =>
        new TransformedTensor {
          val matrix: RealMatrix = {
            val newMatrix = previousTensor.matrix.copy()
            for (i <- offset.indices) {
              newMatrix.addToEntry(i, newMatrix.getColumnDimension - 1, offset(i))
            }
            newMatrix
          }
          val checkpoint: Tensor = previousTensor.checkpoint
          val shape: Seq[Int] = previousTensor.shape
//          val debuggingInformation: Implicitly[DebuggingInformation] = debuggingInformation0
        }
      case _ =>
        new TransformedTensor {
          val checkpoint: Tensor = previousTensor
          val shape: Seq[Int] = checkpoint.shape
//          val debuggingInformation: Implicitly[DebuggingInformation] = debuggingInformation0
          val matrix: RealMatrix = {
            val newMatrix = MatrixUtils.createRealMatrix(shape.length, shape.length + 1)
            for (i <- offset.indices) {
              newMatrix.setEntry(i, i, 1.0)
              newMatrix.setEntry(i, newMatrix.getColumnDimension - 1, offset(i))
            }
            newMatrix
          }
        }
    }
  }

}
