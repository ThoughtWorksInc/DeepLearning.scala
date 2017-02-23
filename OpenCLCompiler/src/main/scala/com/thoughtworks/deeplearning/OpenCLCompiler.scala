package com.thoughtworks.deeplearning

import java.util

import com.dongxiguo.fastring.Fastring
import com.dongxiguo.fastring.Fastring.Implicits._
import com.thoughtworks.deeplearning.OpenCLCompiler.DslFunction.Value
import shapeless.{HList, HNil}

import scala.collection.JavaConverters._
import scala.collection.mutable

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object OpenCLCompiler {

  trait Context {
    def freshName(prefix: String): String
    def getValue(dslFunction: DslFunction[_, _]): Value
    def getPackedType(dslType: DslType[_]): Fastring
  }

  final case class Kernel[Input <: HList, Output](name: String,
                                                  numberOfDimensions: Int,
                                                  dslFunction: DslFunction[Input, Output],
                                                  inputType: DslType[Input],
                                                  outputType: DslType[Output])

  def toSourceCode(kernels: Kernel[_ <: HList, _]*): Fastring = {
    var seed = 0
    def nextId() = {
      val id = seed
      seed += 1
      id
    }
    val types = mutable.Set.empty[Seq[String]]
    val globalDeclarations = mutable.Buffer.empty[Fastring]
    val globalDefinitions = mutable.Buffer.empty[Fastring]

    val exportedFunctions = for {
      Kernel(functionName, numberOfDimensions, dslFunction, inputType, outputType) <- kernels
    } yield {
      val localDefinitions = mutable.Buffer.empty[Fastring]

      val functionCodeCache = new util.IdentityHashMap[DslFunction[_, _], Value]().asScala

      val functionContext = new Context {
        override def getPackedType(dslType: DslType[_]): Fastring = {
          dslType.flatten match {
            case Seq(atom) => fast"$atom"
            case flattenTypes =>
              val identifier = fast"tuple_${flattenTypes.mkFastring("_")}"
              if (!types(flattenTypes)) {
                types += flattenTypes
                globalDeclarations += fast"struct $identifier;"
                globalDefinitions += fastraw"""
struct $identifier {
  ${(for ((t, i) <- flattenTypes.view.zipWithIndex) yield {
                  fast"\n  $t _$i;"
                }).mkFastring}
};"""
              }
              fast"struct $identifier"
          }
        }

        override def getValue(dslFunction: DslFunction[_, _]): Value = {
          functionCodeCache.getOrElseUpdate(
            dslFunction, {
              val code = dslFunction.toCode(this)
              localDefinitions += code.localDefinitions
              globalDeclarations += code.globalDeclarations
              globalDefinitions += code.globalDefinitions
              code.value
            }
          )
        }

        override def freshName(prefix: String): String = {
          raw"""${prefix}_${nextId()}"""
        }
      }
      val indexId = nextId()
      val sizes = for (i <- (1 until numberOfDimensions).reverse.view) yield {
        val nextDimension = i + 1
        if (nextDimension == numberOfDimensions) {
          fast"\n  size_t size${i}_$indexId = get_global_size($i);"
        } else {
          fast"\n  size_t size${i}_$indexId = size$nextDimension * get_global_size($i);"
        }
      }
      val starts = for (i <- (0 until numberOfDimensions).view) yield {
        val nextDimension = i + 1
        if (nextDimension == numberOfDimensions) {
          fast"\n  size_t start${i}_$indexId = get_global_id($i);"
        } else {
          fast"\n  size_t start${i}_$indexId = get_global_id($i) * size$nextDimension;"
        }
      }
      val outputId = (for (i <- (0 until numberOfDimensions).view) yield fast"start${i}_$indexId").mkFastring("+")

      val result = functionContext.getValue(dslFunction)
      val outputTypeName = functionContext.getPackedType(outputType)
      val inputTypeName = functionContext.getPackedType(inputType)

      val inputParameters = for ((inputType, i) <- inputType.flatten.view.zipWithIndex) yield {
        fast"$inputType __input_$i"
      }
      val outputValueName = raw"""output_${nextId()}"""
      val outputParameter = fast"__global $outputTypeName * $outputValueName"
      val allParameters = inputParameters :+ outputParameter
      fastraw"""
__kernel void $functionName(${allParameters.mkFastring(", ")}) {
  ${localDefinitions.mkFastring}
  ${sizes.mkFastring}
  ${starts.mkFastring}
  $outputValueName[$outputId] = ${result.packed};
}
"""
    }
    fastraw"""
${globalDeclarations.mkFastring}
${globalDefinitions.mkFastring}
${exportedFunctions.mkFastring}
"""

  }

  object DslFunction {

    final case class Code(globalDeclarations: Fastring = Fastring.empty,
                          globalDefinitions: Fastring = Fastring.empty,
                          localDefinitions: Fastring = Fastring.empty,
                          value: Value)
    trait Value {
      def unpacked: Seq[Fastring]
      def packed: Fastring
    }

    final case class Atom(value: Fastring) extends Value {
      override def packed: Fastring = value
      override def unpacked = Seq(value)
    }

    final case class Unpacked(unpacked: Seq[Fastring]) extends Value {
      override def packed = fast"{ ${unpacked.mkFastring(", ")} }"
    }

    final case class Packed(packed: Fastring, numberOfFields: Int) extends Value {
      override def unpacked: Seq[Fastring] = {
        for (i <- 0 until numberOfFields) yield {
          fast"$packed._$i"
        }
      }
    }

    final case class DoubleLiteral(data: Double) extends DslFunction[HList, Double] {
      override def toCode(context: Context): Code = {
        Code(value = Atom(value = Fastring(data)))
      }
    }
    final case class FloatLiteral(data: Float) extends DslFunction[HList, Float] {
      override def toCode(context: Context): Code = {
        Code(value = Atom(value = fast"${data}f"))
      }
    }

    final case class Add[Input <: HList, Output](operand1: DslFunction[Input, Output],
                                                 operand2: DslFunction[Input, Output],
                                                 dslType: DslType[Output])
        extends DslFunction[Input, Output] {
      override def toCode(context: Context): Code = {
        val name = context.freshName("plus")
        val flattenType = dslType.flatten
        val packedType = context.getPackedType(dslType)
        Code(
          localDefinitions = fastraw"""
  $packedType $name = ${context.getValue(operand1).packed} + ${context.getValue(operand2).packed};
""",
          value = Packed(Fastring(name), flattenType.length)
        )
      }
    }
  }

  trait DslFunction[-Input <: HList, Output] {

    def toCode(context: Context): DslFunction.Code

  }

  object DslType {

    implicit object DslHNil extends DslType[HNil] {
      override def flatten = Nil
    }

    implicit object DslDouble extends DslType[Double] {
      override val flatten = mutable.WrappedArray.make(Array("double"))
    }

    implicit object DslFloat extends DslType[Float] {
      override val flatten = mutable.WrappedArray.make(Array("float"))
    }

  }

  trait DslType[NativeType] {
    def flatten: Seq[String]
  }
}
