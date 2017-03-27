parallelExecution in Global := false

sbt.dsl.dependsOn(
  DifferentiableBoolean,
  DifferentiableDouble,
  DifferentiableINDArray,
  DifferentiableHList,
  DifferentiableCoproduct,
  DifferentiableSeq,
  DifferentiableAny,
  DifferentiableNothing
)

lazy val Layer = project

lazy val Symbolic = project.dependsOn(Layer)

lazy val DifferentiableBoolean = project.dependsOn(Layer, CumulativeLayer, Poly)

lazy val DifferentiableDouble =
  project.dependsOn(Poly, DifferentiableBoolean, CumulativeLayer, DifferentiableAny)

lazy val DifferentiableFloat =
  project.dependsOn(Poly, DifferentiableBoolean, CumulativeLayer, DifferentiableAny)

val DoubleRegex = """(?i:double)""".r

sourceGenerators in Compile in DifferentiableFloat += Def.task {
  for {
    doubleFile <- (unmanagedSources in Compile in DifferentiableDouble).value
    relativeFile <- doubleFile.relativeTo((sourceDirectory in Compile in DifferentiableDouble).value)
  } yield {
    val doubleSource = IO.read(doubleFile, scala.io.Codec.UTF8.charSet)

    val floatSource = DoubleRegex.replaceAllIn(doubleSource, { m =>
      m.matched match {
        case "Double" => "Float"
        case "double" => "float"
      }
    })

    val outputFile = (sourceManaged in Compile in DifferentiableFloat).value / relativeFile.getPath
    IO.write(outputFile, floatSource, scala.io.Codec.UTF8.charSet)
    outputFile
  }
}.taskValue

lazy val DifferentiableInt =
  project.dependsOn(Poly, DifferentiableDouble, DifferentiableBoolean, CumulativeLayer, DifferentiableAny)

lazy val Poly = project.dependsOn(Symbolic)

lazy val DifferentiableAny = project.dependsOn(Symbolic)

lazy val DifferentiableNothing = project.dependsOn(Symbolic)

lazy val DifferentiableSeq = project.dependsOn(DifferentiableInt)

lazy val DifferentiableINDArray =
  project.dependsOn(DifferentiableInt, DifferentiableDouble, DifferentiableSeq)

lazy val DifferentiableHList = project.dependsOn(Poly)

lazy val DifferentiableCoproduct = project.dependsOn(DifferentiableBoolean)

lazy val CumulativeLayer = project.dependsOn(Layer)

addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0" % Test

addCompilerPlugin("com.thoughtworks.implicit-dependent-type" %% "implicit-dependent-type" % "2.0.0" % Test)

libraryDependencies += "com.thoughtworks.enableIf" %% "enableif" % "1.1.4" % Test

libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "0.7.2" % Test

crossScalaVersions := Seq("2.10.6", "2.11.8", "2.12.1")

publishArtifact := false

lazy val unidoc = project
  .enablePlugins(TravisUnidocTitle)
  .settings(
    UnidocKeys.unidocProjectFilter in ScalaUnidoc in UnidocKeys.unidoc := {
      import Ordering.Implicits._
      if (VersionNumber(scalaVersion.value).numbers >= Seq(2, 12)) {
        inAnyProject -- inProjects(DifferentiableINDArray)
      } else {
        inAnyProject
      }
    },
    addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)
  )


organization in ThisBuild := "com.thoughtworks.deeplearning"

fork in Test := true
