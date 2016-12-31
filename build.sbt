sbt.dsl.dependsOn(DifferentiableBoolean,
                  DifferentiableDouble,
                  DifferentiableINDArray,
                  DifferentiableHList,
                  DifferentiableCoproduct,
                  DifferentiableSeq,
                  DifferentiableAny,
                  DifferentiableNothing)

lazy val Layer = project.disablePlugins(SparkPackagePlugin)

lazy val Lift = project.disablePlugins(SparkPackagePlugin).dependsOn(Layer)

lazy val DifferentiableBoolean = project.disablePlugins(SparkPackagePlugin).dependsOn(Layer, BufferedLayer, Poly)

lazy val DifferentiableDouble =
  project.disablePlugins(SparkPackagePlugin).dependsOn(Poly, DifferentiableBoolean, BufferedLayer, DifferentiableAny)

lazy val DifferentiableFloat =
  project.disablePlugins(SparkPackagePlugin).dependsOn(Poly, DifferentiableBoolean, BufferedLayer, DifferentiableAny)

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

lazy val DifferentiableInt = project
  .disablePlugins(SparkPackagePlugin)
  .dependsOn(Poly, DifferentiableDouble, DifferentiableBoolean, BufferedLayer, DifferentiableAny)

lazy val Poly = project.disablePlugins(SparkPackagePlugin).dependsOn(Lift)

lazy val DifferentiableAny = project.disablePlugins(SparkPackagePlugin).dependsOn(Lift)

lazy val DifferentiableNothing = project.disablePlugins(SparkPackagePlugin).dependsOn(Lift)

lazy val DifferentiableSeq = project.disablePlugins(SparkPackagePlugin).dependsOn(DifferentiableInt)

lazy val DifferentiableINDArray =
  project.disablePlugins(SparkPackagePlugin).dependsOn(DifferentiableInt, DifferentiableDouble)

lazy val DifferentiableHList = project.disablePlugins(SparkPackagePlugin).dependsOn(Poly)

lazy val DifferentiableCoproduct = project.disablePlugins(SparkPackagePlugin).dependsOn(DifferentiableBoolean)

lazy val BufferedLayer = project.disablePlugins(SparkPackagePlugin).dependsOn(Layer)

addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0" % Test

addCompilerPlugin("com.thoughtworks.implicit-dependent-type" %% "implicit-dependent-type" % "1.0.0" % Test)

crossScalaVersions := Seq("2.10.6", "2.11.8")

publishArtifact := false

lazy val unidoc = project
  .enablePlugins(TravisUnidocTitle)
  .settings(addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full))

organization in ThisBuild := "com.thoughtworks.deeplearning"

fork in Test := true
