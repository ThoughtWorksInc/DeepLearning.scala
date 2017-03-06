package com.thoughtworks.deeplearning

import java.util

import com.dongxiguo.fastring.Fastring
import com.dongxiguo.fastring.Fastring.Implicits._
import com.thoughtworks.deeplearning.OpenCLCompiler.DslFunction.{Unpacked, Value}
import com.thoughtworks.deeplearning.OpenCLCompiler.DslType.HListType
import shapeless._
import shapeless.ops.hlist.LiftAll

import scala.collection.JavaConverters._
import scala.collection.immutable.Queue
import scala.collection.{SeqView, mutable}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object OpenCLCompiler {

  trait Context {
    def freshName(prefix: String): String
    def getValue(dslFunction: DslFunction[_, _]): Value
    def getPackedType(dslType: DslType[_]): Fastring
  }

  // TODO: Turn Kernel to a trait
  final case class Kernel[Input <: HList, Output <: HList](name: String,
                                                           numberOfDimensions: Int,
                                                           dslFunction: DslFunction[Input, Output],
                                                           inputType: HListType[Input],
                                                           outputType: HListType[Output])

  def toSourceCode(kernels: Kernel[_ <: HList, _ <: HList]*): Fastring = {
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

      // TODO: reference to input
      val result = functionContext.getValue(dslFunction)

      val inputParameters = inputType.fieldTypes.view.zipWithIndex.foldLeft[Queue[Fastring]](Queue.empty) {
        case (inputParameters, (inputFieldType, i)) =>
          if (inputFieldType.flatten.isEmpty) {
            inputParameters
          } else {
            val inputValueName = raw"""input_${nextId()}"""
            val inputTypeName = functionContext.getPackedType(inputFieldType)
            inputParameters.enqueue(fast"$inputTypeName $inputValueName")
          }
      }

      val (outputParameters, setters, _) = outputType.fieldTypes.view.zipWithIndex
        .foldLeft[(Queue[Fastring], Queue[Fastring], SeqView[Fastring, Seq[_]])](Queue.empty,
                                                                                 Queue.empty,
                                                                                 result.unpacked.view) {
          case ((outputParameters, setters, unpackedResults), (outputFieldType, i)) =>
            outputFieldType.flatten.length match {
              case 0 =>
                (outputParameters, setters, unpackedResults)
              case fieldSize =>
                val outputValueName = raw"""output_${nextId()}"""
                val outputTypeName = functionContext.getPackedType(outputFieldType)
                val outputParameter = fast"__global $outputTypeName * $outputValueName"

                val (unpackedField, restUnpackedResults) = unpackedResults.splitAt(fieldSize)

                val returnValueName = raw"""return_${nextId()}"""
                val setter = fast"""
  ${functionContext.getPackedType(outputFieldType)} $returnValueName = ${Unpacked(unpackedField.force).packed};
  $outputValueName[$outputId] = $returnValueName;
"""
                (outputParameters.enqueue(outputParameter), setters.enqueue(setter), restUnpackedResults)
            }
        }

      val allParameters = inputParameters ++ outputParameters
      fastraw"""
__kernel void $functionName(${allParameters.mkFastring(", ")}) {
  ${localDefinitions.mkFastring}
  ${sizes.mkFastring}
  ${starts.mkFastring}
  ${setters.mkFastring}
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
      override def packed = unpacked match {
        case Seq(single) => single
        case _ => fast"{ ${unpacked.mkFastring(", ")} }"
      }
    }

    final case class Packed(packed: Fastring, numberOfFields: Int) extends Value {
      override def unpacked: Seq[Fastring] = {
        if (numberOfFields == 1) {
          Seq(packed)
        } else {
          for (i <- 0 until numberOfFields) yield {
            fast"$packed._$i"
          }
        }
      }
    }

    final case object HNilLiteral extends DslFunction[HList, HNil] {
      override def toCode(context: Context): Code = {
        Code(value = Unpacked(Nil))
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

    final case class HCons[Input <: HList, Head, Tail <: HList](operand1: DslFunction[Input, Head],
                                                                operand2: DslFunction[Input, Tail])
        extends DslFunction[Input, Head :: Tail] {
      override def toCode(context: Context): Code = {
        Code(
          value = Unpacked(context.getValue(operand1).unpacked ++ context.getValue(operand2).unpacked)
        )
      }
    }

  }

  trait DslFunction[-Input <: HList, Output] {

    def toCode(context: Context): DslFunction.Code

  }

  object DslType {

    implicit def dslHCons[Head, Tail <: HList](implicit headType: DslType[Head],
                                               tailType: HListType[Tail]): HListType[Head :: Tail] = {
      new HListType[Head :: Tail] {

        override val flatten: Stream[String] = {
          fieldTypes.toStream.flatMap(_.flatten)
        }

        override def fieldTypes = {
          headType :: tailType.fieldTypes
        }
      }
    }

    implicit object DslHNil extends HListType[HNil] {
      override def flatten = Nil

      override def fieldTypes = Nil
    }

    implicit object DslDouble extends DslType[Double] {
      override val flatten = Seq("double")
    }

    implicit object DslFloat extends DslType[Float] {
      override val flatten = Seq("float")
    }

    trait HListType[NativeType <: HList] extends DslType[NativeType] {
      def fieldTypes: List[DslType[_]]

    }

  }

  trait DslType[NativeType] {
    def flatten: Seq[String]

  }

}
