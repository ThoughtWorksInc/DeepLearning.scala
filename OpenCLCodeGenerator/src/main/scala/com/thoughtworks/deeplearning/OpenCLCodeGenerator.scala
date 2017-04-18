package com.thoughtworks.deeplearning

import java.util

import com.dongxiguo.fastring.Fastring
import com.dongxiguo.fastring.Fastring.Implicits._
import shapeless._

import scala.collection.JavaConverters._
import scala.collection.immutable.Queue
import scala.collection.{SeqView, mutable}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object OpenCLCodeGenerator { // TODO: rename to OpenCLCodeGenerator

  trait Context {
    def freshName(prefix: String): String

    def resolve(id: Any): DslExpression.Accessor

    def get(dslFunction: DslExpression): DslExpression.Accessor
    def get(dslType: DslType): DslType.Accessor
    def get(effect: DslEffect): DslEffect.Statement
  }

  final case class Parameter(id: Any, dslType: DslType)

  final case class KernelDefinition(name: String, parameters: Seq[Parameter], effects: Seq[DslEffect])

  def generateSourceCode(kernels: KernelDefinition*): Fastring = {
    var seed = 0
    def nextId() = {
      val id = seed
      seed += 1
      id
    }

    val types = mutable.Set.empty[Seq[String]]
    val globalDeclarations = mutable.Buffer.empty[Fastring]
    val globalDefinitions = mutable.Buffer.empty[Fastring]
    val typeCodeCache = mutable.HashMap.empty[DslType, DslType.Accessor]

    val exportedFunctions = for {
      KernelDefinition(functionName, parameters, effects) <- kernels
    } yield {

      val parameterMap = mutable.Map.empty[Any, (String, DslType)]

      val localDefinitions = mutable.Buffer.empty[Fastring]

      val expressionCodeCache = new util.IdentityHashMap[DslExpression, DslExpression.Accessor]().asScala
      val effectCodeCache = new util.IdentityHashMap[DslEffect, Fastring]().asScala

      val functionContext = new Context {

        override def get(dslType: DslType): DslType.Accessor = {
          typeCodeCache.getOrElseUpdate(dslType, {
            val code = dslType.toCode(this)
            globalDeclarations += code.globalDeclarations
            globalDefinitions += code.globalDefinitions
            code.accessor

          })
        }
        override def get(expression: DslExpression): DslExpression.Accessor = {
          expressionCodeCache.getOrElseUpdate(
            expression, {
              val code = expression.toCode(this)
              localDefinitions += code.localDefinitions
              globalDeclarations += code.globalDeclarations
              globalDefinitions += code.globalDefinitions
              code.accessor
            }
          )
        }

        override def freshName(prefix: String): String = {
          raw"""${prefix}_${nextId()}"""
        }

        override def get(effect: DslEffect): Fastring = {
          effectCodeCache.getOrElseUpdate(
            effect, {
              val code = effect.toCode(this)
              localDefinitions += code.localDefinitions
              globalDeclarations += code.globalDeclarations
              globalDefinitions += code.globalDefinitions
              code.statements
            }
          )
        }

        override def resolve(id: Any) = {
          val (name, dslType) = parameterMap(id)
          DslExpression.Accessor.Packed(fast"$name", get(dslType).unpacked.length)
        }
      }

      val parameterDeclarations = for (parameter <- parameters) yield {
        val name = s"parameter_${nextId()}"
        parameterMap(parameter.id) = name -> parameter.dslType
        val typeName = functionContext.get(parameter.dslType).packed
        fast"$typeName $name"
      }

      val effectStatements = for (effect <- effects) yield {
        functionContext.get(effect)
      }

      fastraw"""
__kernel void $functionName(${parameterDeclarations.mkFastring(", ")}) {
  ${localDefinitions.mkFastring}
  ${effectStatements.mkFastring}
}
"""
    }
    fastraw"""
${globalDeclarations.mkFastring}
${globalDefinitions.mkFastring}
${exportedFunctions.mkFastring}
"""

  }

  object DslExpression {

    final case class Code(globalDeclarations: Fastring = Fastring.empty,
                          globalDefinitions: Fastring = Fastring.empty,
                          localDefinitions: Fastring = Fastring.empty,
                          accessor: Accessor)
    trait Accessor {
      def unpacked: Seq[Fastring]
      def packed: Fastring
    }

    object Accessor {

      final case class Atom(value: Fastring) extends Accessor {
        override def packed: Fastring = value
        override def unpacked = Seq(value)
      }

      final case class Unpacked(unpacked: Seq[Fastring]) extends Accessor {
        override def packed = unpacked match {
          case Seq(single) => single
          case _ => fast"{ ${unpacked.mkFastring(", ")} }"
        }
      }

      final case class Packed(packed: Fastring, numberOfFields: Int) extends Accessor {
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
    }

    import Accessor._

    final case object HNilLiteral extends DslExpression {
      override def toCode(context: Context): Code = {
        Code(accessor = Unpacked(Nil))
      }
    }

    final case class DoubleLiteral(data: Double) extends DslExpression {
      override def toCode(context: Context): Code = {
        Code(accessor = Atom(value = Fastring(data)))
      }
    }

    final case class IntLiteral(data: Int) extends DslExpression {
      override def toCode(context: Context): Code = {
        Code(accessor = Atom(value = Fastring(data)))
      }
    }

    final case class FloatLiteral(data: Float) extends DslExpression {
      override def toCode(context: Context): Code = {
        Code(accessor = Atom(value = fast"${data}f"))
      }
    }

    final case class Identifier(id: Any) extends DslExpression {
      override def toCode(context: Context): Code = {
        Code(accessor = context.resolve(id))
      }
    }

    final case class GetGlobalId(dimensionIndex: DslExpression) extends DslExpression {
      override def toCode(context: Context): Code = {
        val i = context.get(dimensionIndex)
        Code(
          accessor = Atom(fast"get_global_id(${i.packed})")
        )

      }
    }

    final case class Plus(operand1: DslExpression, operand2: DslExpression, dslType: DslType) extends DslExpression {
      override def toCode(context: Context): Code = {
        val name = context.freshName("plus")
        val typeReference = context.get(dslType)
        val packedType = typeReference.packed
        Code(
          localDefinitions = fastraw"""
  $packedType $name = ${context.get(operand1).packed} + ${context.get(operand2).packed};""",
          accessor = Packed(Fastring(name), typeReference.unpacked.length)
        )
      }
    }

    final case class Times(operand1: DslExpression, operand2: DslExpression, dslType: DslType) extends DslExpression {
      override def toCode(context: Context): Code = {
        val name = context.freshName("plus")
        val typeReference = context.get(dslType)
        val packedType = typeReference.packed
        Code(
          localDefinitions = fastraw"""
  $packedType $name = ${context.get(operand1).packed} * ${context.get(operand2).packed};""",
          accessor = Packed(Fastring(name), typeReference.unpacked.length)
        )
      }
    }

    final case class HCons(operand1: DslExpression, operand2: DslExpression) extends DslExpression {
      override def toCode(context: Context): Code = {
        Code(
          accessor = Unpacked(context.get(operand1).unpacked ++ context.get(operand2).unpacked)
        )
      }
    }

  }

  trait DslExpression {

    def toCode(context: Context): DslExpression.Code

  }

  trait DslEffect {

    def toCode(context: Context): DslEffect.Code

  }

  object DslEffect {

    type Statement = Fastring

    final case class Code(globalDeclarations: Fastring = Fastring.empty,
                          globalDefinitions: Fastring = Fastring.empty,
                          localDefinitions: Fastring = Fastring.empty,
                          statements: Fastring = Fastring.empty)

    final case class Update(buffer: DslExpression, index: DslExpression, value: DslExpression, valueType: DslType)
        extends DslEffect {
      override def toCode(context: Context): Code = {
        val valueName = context.freshName("update")
        Code(
          localDefinitions = fast"""
  ${context.get(valueType).packed} $valueName = ${context.get(value).packed};""",
          statements = fast"""
  ${context.get(buffer).packed}[${context.get(index).packed}] = $valueName;"""
        )
      }
    }

  }

  object DslType {

    trait Accessor {
      def packed: Fastring
      def unpacked: Seq[String]
    }

    object Accessor {
      final case class Structure(name: String, override val unpacked: Seq[String]) extends Accessor {
        override def packed: Fastring = fast"struct $name"
      }

      final case class Atom(name: String) extends Accessor {
        override def packed: Fastring = fast"$name"

        override def unpacked: Seq[String] = Seq(name)
      }
    }
    import Accessor._
    final case class Code(globalDeclarations: Fastring = Fastring.empty,
                          globalDefinitions: Fastring = Fastring.empty,
                          accessor: Accessor)

    final case class DslStructure(fieldTypes: List[DslType]) extends DslType {
      override def toCode(context: Context): DslType.Code = {
        fieldTypes match {
          case head +: Nil =>
            Code(accessor = context.get(head))
          case _ =>
            val name = context.freshName("struct")
            val flatten = fieldTypes.flatMap { fieldType =>
              context.get(fieldType).unpacked
            }
            Code(
              globalDeclarations = fast"struct $name;",
              globalDefinitions = fastraw"""
struct $name {
  ${(for ((t, i) <- flatten.view.zipWithIndex) yield fast"\n  $t _$i;").mkFastring}
};""",
              Structure(name, flatten)
            )
        }
      }
    }

    case object DslDouble extends DslType {
      override def toCode(context: Context): Code =
        Code(accessor = Atom("double"))
    }

    case object DslFloat extends DslType {
      override def toCode(context: Context): Code =
        Code(accessor = Atom("float"))
    }

    case object DslInt extends DslType {
      override def toCode(context: Context): Code =
        Code(accessor = Atom("int"))
    }

    final case class DslBuffer(elementType: DslType) extends DslType {

      override def toCode(context: Context): Code = {
        val name = context.freshName("buffer")
        val element = context.get(elementType)
        Code(
          globalDeclarations = Fastring.empty,
          globalDefinitions = fast"typedef global ${element.packed} * $name;",
          Atom(name)
        )
      }
    }

  }

  trait DslType {

    def toCode(context: Context): DslType.Code

  }

}
