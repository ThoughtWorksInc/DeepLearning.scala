package com.thoughtworks.deeplearning

import java.util

import com.dongxiguo.fastring.Fastring
import com.dongxiguo.fastring.Fastring.Implicits._
import shapeless.{HList, HNil}

import scala.collection.JavaConverters._
import scala.collection.mutable

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object OpenCL {

  trait Context {
    def freshName(prefix: String): String
    def getValue(dslFunction: DslFunction[_, _]): Fastring
    def getType(dslType: DslType[_]): Fastring
  }

  def compile(dslFunctions: Map[String, TypedFunction[_, _]]): Fastring = {
    var seed = 0
    def nextId() = {
      val id = seed
      seed += 1
      id
    }
    val types = mutable.Map.empty[String, Fastring]
    val globalDeclarations = mutable.Buffer.empty[Fastring]
    val globalDefinitions = mutable.Buffer.empty[Fastring]

    val exportedFunctions = for ((functionName, TypedFunction(dslFunction, inputType, outputType)) <- dslFunctions)
      yield {
        val localDefinitions = mutable.Buffer.empty[Fastring]

        val functionCodeCache = new util.IdentityHashMap[DslFunction[_, _], Fastring]().asScala

        val functionContext = new Context {
          override def getType(dslType: DslType[_]): Fastring = {
            val code = dslType.toCode(this)
            types.get(code.identifier) match {
              case Some(typeName) => typeName
              case None =>
                globalDeclarations += code.globalDeclarations
                globalDefinitions += code.globalDefinitions
                types.put(code.identifier, code.typeName)
                code.typeName
            }
          }

          override def getValue(dslFunction: DslFunction[_, _]): Fastring = {
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

        val result = functionContext.getValue(dslFunction)
        val outputTypeName = functionContext.getType(outputType)
        val inputTypeName = functionContext.getType(inputType)
        val outputValueName = raw"""output_${nextId()}"""
        fastraw"""
static size_t __output_id() {
  uint rank = get_work_dim();
  uint i = rank;
  size_t skip = 1;
  size_t result = 0;
  do {
    i--;
    result += skip * get_global_id(i);
    skip *= get_global_size(i);
  } while(i != 0);
  return result;
}

__kernel void $functionName(/*$inputTypeName input, */ __global $outputTypeName * $outputValueName) {
  ${localDefinitions.mkFastring}
  $outputValueName[__output_id()] = $result;
}
"""
      }
    fastraw"""
${globalDeclarations.mkFastring}
${globalDefinitions.mkFastring}
${exportedFunctions.mkFastring}
"""

  }

  final case class TypedFunction[Input <: HList, Output](dslFunction: DslFunction[Input, Output],
                                                         inputType: DslType[Input],
                                                         outputType: DslType[Output])

  object DslFunction {

    final case class Code(globalDeclarations: Fastring = Fastring.empty,
                          globalDefinitions: Fastring = Fastring.empty,
                          localDefinitions: Fastring = Fastring.empty,
                          value: Fastring)
    final case class DoubleLiteral(data: Double) extends DslFunction[HList, Double] {
      override def toCode(context: Context): Code = {
        Code(value = Fastring(data))
      }
    }
    final case class FloatLiteral(data: Float) extends DslFunction[HList, Float] {
      override def toCode(context: Context): Code = {
        Code(value = fast"${data}f")
      }
    }

    final case class Add[Input <: HList, Output](operand1: DslFunction[Input, Output],
                                                 operand2: DslFunction[Input, Output],
                                                 dslType: DslType[Output])
        extends DslFunction[Input, Output] {
      override def toCode(context: Context): Code = {
        val name = context.freshName("plus")
        Code(
          localDefinitions = fastraw"""
  ${context.getType(dslType)} $name = ${context.getValue(operand1)} + ${context.getValue(operand2)};
""",
          value = Fastring(name)
        )
      }
    }
  }

  trait DslFunction[-Input <: HList, Output] {

    def toCode(context: Context): DslFunction.Code

  }

  object DslType {
    final case class Code(globalDeclarations: Fastring = Fastring.empty,
                          globalDefinitions: Fastring = Fastring.empty,
                          identifier: String,
                          typeName: Fastring)

    object Code {
      def builtInType(identifier: String) = {
        new Code(identifier = identifier, typeName = Fastring(identifier))
      }

      def structType(definitionBody: Fastring, identifier: String) = {
        new Code(
          globalDeclarations = fast"struct $identifier;",
          globalDefinitions = fast"""
struct $identifier {
  $definitionBody
};""",
          identifier = identifier,
          typeName = fast"struct $identifier"
        )

      }
    }

    object DslHNil extends DslType[HNil] {
      override def toCode(context: Context): Code = {
        Code.structType(fast"", "HNil")
      }
    }

    object DslDouble extends DslType[Double] {
      override def toCode(context: Context): Code = {
        Code.builtInType("double")

      }
    }
    object DslFloat extends DslType[Float] {
      override def toCode(context: Context): Code = {
        Code.builtInType("float")
      }
    }

  }

  trait DslType[NativeType] {
    def toCode(context: Context): DslType.Code
  }
}
