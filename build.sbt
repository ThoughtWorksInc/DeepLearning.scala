parallelExecution in Global := false

lazy val DifferentiableKernel =
  project.dependsOn(OpenCL, OpenCLCodeGenerator, CumulativeTape, Layer, `stateless-future-util`, Symbolic % Test)

lazy val OpenCLCodeGenerator = project.dependsOn(Memory)

// TODO: Create a separate Tape library?
lazy val Layer = project.dependsOn(`stateless-future`, `stateless-future-scalatest` % Test)

lazy val CumulativeTape = project.dependsOn(Layer)

lazy val CheckedTape = project.dependsOn(Layer, Closeables)

lazy val Memory = project

lazy val Closeables = project.dependsOn(`stateless-future-sde`, `stateless-future-scalatest` % Test)

// TODO: Rename to ToLiteral?
lazy val Symbolic = project.dependsOn(Layer)

lazy val `stateless-future` = project

lazy val `stateless-future-scalaz` = project.dependsOn(`stateless-future`, `stateless-future-scalatest` % Test)

lazy val `stateless-future-sde` = project.dependsOn(`stateless-future-scalaz`, `stateless-future-scalatest` % Test)

lazy val `stateless-future-scalatest` = project.dependsOn(`stateless-future-util`)

lazy val `stateless-future-util` = project.dependsOn(`stateless-future`)

lazy val OpenCL = project.dependsOn(Closeables, `stateless-future`, Memory)

lazy val LayerFactory = project.dependsOn(DifferentiableKernel)

lazy val DifferentiableFloat =
  project.dependsOn(Layer, CumulativeTape, CheckedTape, `stateless-future-scalatest` % Test, Symbolic % Test)

lazy val DifferentiableDouble =
  project.dependsOn(Layer, CumulativeTape, CheckedTape, `stateless-future-scalatest` % Test, Symbolic % Test)

lazy val DifferentiableInt =
  project.dependsOn(Layer,
                    CumulativeTape,
                    CheckedTape,
                    DifferentiableFloat,
                    `stateless-future-util` % Test,
                    Symbolic % Test)

val FloatRegex = """(?i:float)""".r

sourceGenerators in Compile in DifferentiableDouble += Def.task {
  for {
    floatFile <- (unmanagedSources in Compile in DifferentiableFloat).value
    relativeFile <- floatFile.relativeTo((sourceDirectory in Compile in DifferentiableFloat).value)
  } yield {
    val floatSource = IO.read(floatFile, scala.io.Codec.UTF8.charSet)

    val doubleSource = FloatRegex.replaceAllIn(floatSource, { m =>
      m.matched match {
        case "Float" => "Double"
        case "float" => "double"
      }
    })

    val outputFile = (sourceManaged in Compile in DifferentiableDouble).value / relativeFile.getPath
    IO.write(outputFile, doubleSource, scala.io.Codec.UTF8.charSet)
    outputFile
  }
}.taskValue

crossScalaVersions := Seq("2.11.8", "2.12.1")

publishArtifact := false

lazy val unidoc = project
  .enablePlugins(TravisUnidocTitle)
  .settings(
    addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full),
    scalaOrganization in updateSbtClassifiers := (scalaOrganization in Global).value,
    scalaOrganization := "org.typelevel",
    scalacOptions += "-Yliteral-types"
  )

organization in ThisBuild := "com.thoughtworks.deeplearning"

fork in Test := true
